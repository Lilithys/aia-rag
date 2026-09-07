"""Consolidate the frozen final holdout without making any API calls."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
import math

from src.artifacts import file_sha256
from src.evaluation.paid_calls_v1 import BudgetLedger, atomic_json


def percentile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def ledger_summary(path: Path, cap: float) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing usage ledger; cannot interpret absent billing as zero: {path}")
    ledger = BudgetLedger(path, cap)
    receipts = ledger.receipts()
    usage = [row["usage"] for row in receipts if not row.get("usage_unknown", True)]
    return {
        "attempts": len(receipts),
        "accounted_rmb": ledger.spent_upper(),
        "cap_rmb": cap,
        "unknown_usage_attempts": sum(row.get("usage_unknown", True) for row in receipts),
        "prompt_tokens": sum(row["prompt_tokens"] for row in usage),
        "completion_tokens": sum(row["completion_tokens"] for row in usage),
        "cached_prompt_tokens": sum(row["cached_tokens"] for row in usage),
        "first_attempt_utc": min((row["time_utc"] for row in receipts), default=None),
        "last_attempt_utc": max((row["time_utc"] for row in receipts), default=None),
    }


def threshold_status(bounds, target):
    if bounds[0] >= target:
        return "met"
    if bounds[1] < target:
        return "not_met"
    return "indeterminate_due_to_unknown_scores"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the stored analysis without rewriting it.",
    )
    args = parser.parse_args()
    root = args.root
    generation_plan = json.loads((root / "plan.json").read_text())
    bundle = json.loads((root / "generation_bundle.json").read_text())
    generation_summary = json.loads((root / "generation_summary.json").read_text())
    reports = {name: json.loads((root / name / "report.json").read_text())
               for name in ("correctness", "faithfulness", "context_precision")}
    candidate = bundle["candidates"][0]
    metric = {
        "correctness": reports["correctness"]["summaries"][candidate]["overall"]["correctness"],
        "faithfulness": reports["faithfulness"]["summaries"][candidate]["overall"]["faithfulness"],
        "context_precision": reports["context_precision"]["summaries"][candidate]["overall"]["context_precision"],
    }
    totals = [case["latency_ms"]["total"] for case in bundle["cases"] if case["latency_ms"]]
    successful = [case["latency_ms"]["total"] for case in bundle["cases"]
                  if case["status"] == "ok" and case["latency_ms"]]
    submitted = sum(case["status"] != "not_submitted" for case in bundle["cases"])
    success_within = sum(case["status"] == "ok" and case["latency_ms"]
                         and case["latency_ms"]["total"] <= 10_000 for case in bundle["cases"])
    generation_cost = ledger_summary(root / "runtime/usage.jsonl", generation_plan["binding"]["generation_budget_cap_rmb"])
    scoring_costs = {}
    for name in reports:
        score_plan = json.loads((root / name / "plan.json").read_text())
        scoring_costs[name] = ledger_summary(root / name / "usage.jsonl", score_plan["binding"]["budget_cap_rmb"])
    total_cost = generation_cost["accounted_rmb"] + sum(row["accounted_rmb"] for row in scoring_costs.values())
    correctness_bounds = metric["correctness"]["bounds"]
    faith_bounds = metric["faithfulness"]["conservative_bounds_allowing_missing_to_be_fact_bearing"]
    cp_bounds = metric["context_precision"]["bounds"]
    latency_rate = success_within / 100
    result = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split": bundle["split"],
        "candidate": candidate,
        "generation": generation_summary,
        "metrics": metric,
        "acceptance": {
            "correctness_gte_0_80": threshold_status(correctness_bounds, 0.80),
            "faithfulness_gte_0_85": threshold_status(faith_bounds, 0.85),
            "context_precision_gte_0_70": threshold_status(cp_bounds, 0.70),
            "success_within_10s_gte_0_90": "met" if latency_rate >= 0.90 else "not_met",
        },
        "latency": {
            "denominator_all_planned": 100,
            "submitted": submitted,
            "successful": len(successful),
            "successful_within_10s": success_within,
            "success_within_10s_over_all_planned": latency_rate,
            "p50_success_seconds": statistics.median(successful) / 1000 if successful else None,
            "p90_success_seconds": percentile(successful, 0.9) / 1000 if successful else None,
            "max_submitted_seconds": max(totals) / 1000 if totals else None,
            "scope": "Warm final runtime requests; includes queue, retrieval, reranking, generation, citation validation and logging. Process/model cold start is separate.",
        },
        "costs": {
            "generation": generation_cost,
            "scoring": scoring_costs,
            "total_final_holdout_rmb": total_cost,
            "observed_generation_rmb_per_1000_submitted": generation_cost["accounted_rmb"] / submitted * 1000 if submitted else None,
            "scope": "API usage only. Local CPU/MPS, storage, OCR, embedding and reranker compute are not monetized.",
        },
        "scoring_completion": {name: {
            "completed": report["completed_metric_tasks"],
            "successful": report["successful_metric_tasks"],
            "pending": report["pending_metric_tasks"],
        } for name, report in reports.items()},
        "bindings": {
            "generation": generation_plan["binding_sha256"],
            **{name: json.loads((root / name / "plan.json").read_text())["binding_sha256"] for name in reports},
        },
        "source_sha256": {str(path): file_sha256(path) for path in (
            root / "plan.json", root / "generation_bundle.json",
            root / "correctness/report.json", root / "faithfulness/report.json",
            root / "context_precision/report.json",
        )},
        "limitations": [
            "Correctness v1.3 is an automated proxy with known diagnostic gaps and no independent human calibration.",
            "Unknown judge results remain unknown; they are not silently dropped or counted as passes.",
            "The holdout was evaluated once after configuration freeze and cannot be used for further tuning.",
            "Recorded-history questions are not closed-loop conversations driven by the system's own earlier answers.",
        ],
    }
    if bundle["split"] != "holdout" or len(bundle["cases"]) != 100:
        raise ValueError("Final analysis requires the 100-question holdout")
    if submitted != 100 or any(report["pending_metric_tasks"] for report in reports.values()):
        raise ValueError("Holdout generation or scoring is incomplete")
    analysis_path = root / "analysis.json"
    if args.check:
        stored = json.loads(analysis_path.read_text())
        comparable_result = {key: value for key, value in result.items() if key != "created_utc"}
        comparable_stored = {key: value for key, value in stored.items() if key != "created_utc"}
        if comparable_result != comparable_stored:
            raise ValueError("Stored final analysis does not match its source artifacts")
    else:
        atomic_json(analysis_path, result)
    print(json.dumps({"acceptance": result["acceptance"], "costs": result["costs"],
                      "latency": result["latency"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
