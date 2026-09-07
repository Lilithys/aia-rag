"""Versioned v1.1 offline plan/report and separately invoked budgeted judging.

Consumes a complete generation bundle; no retriever or generator is started.
The bundle contract supports smoke/dev/holdout. Legacy import is smoke-only.
No calibration/acceptance claim follows merely from executing the judges.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

from src.artifacts import file_sha256, fingerprint
from src.data_prep.revise_eval_v11 import validate_revision
from .paid_calls_v1 import BudgetLedger, CallConfig, JsonCaller, PRICING, atomic_json, input_upper_bound, price
from .prepare_calibration_v11 import build_pack, read_record
from .prompts_v1 import validate_correctness, validate_generation
from .rubric_v11 import correctness_messages, faithfulness_messages, score_units

METRICS = ("correctness", "faithfulness", "context_precision")


def import_legacy(parent, release, run, target):
    if target.exists():
        raise FileExistsError("Generation bundle exists")
    audited = build_pack(parent, release, run)
    cases = []
    for row in audited["generation_reuse"]:
        generated = read_record(Path(row["generation_file"]))
        cases.append({"candidate": row["candidate"], "question_id": row["question_id"],
            "item_sha256": row["new_item_sha256"], "status": generated["status"],
            "response": generated["response"], "contexts": generated["contexts"],
            "retrieval_contexts": generated["contexts"] if generated.get("gate") is None and "retrieval" in generated else None,
            "source_generation_file": row["generation_file"], "source_generation_file_sha256": row["generation_file_sha256"],
            "latency_origin": "v1.0 original run; not remeasured", "generation_reuse_proof": row})
    plan = json.loads((run / "plan.json").read_text())
    bundle = {"format": "generation-bundle-v1.1", "split": "smoke",
              "release_manifest_sha256": file_sha256(release / "release_manifest.json"),
              "candidates": list(plan["binding"]["candidates"]), "generation_config": plan["binding"],
              "origin": "legacy v1.0 smoke responses, input-equivalence audited; not final service outputs",
              "cases": cases}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"imported": len(cases), "failed": sum(c["status"] != "ok" for c in cases), "api_calls": 0}


def validate_contexts(contexts):
    if not isinstance(contexts, list) or any(not isinstance(c, dict) or set(c) != {"chunk_id", "text"}
        or not isinstance(c["chunk_id"], str) or not c["chunk_id"] or not isinstance(c["text"], str) for c in contexts):
        raise ValueError("Invalid context records")
    if len({c["chunk_id"] for c in contexts}) != len(contexts):
        raise ValueError("Duplicate retrieved chunk IDs")


def load_bundle(parent, release, path):
    validate_revision(parent, release)
    bundle = json.loads(path.read_text())
    if bundle["format"] != "generation-bundle-v1.1" or bundle["release_manifest_sha256"] != file_sha256(release / "release_manifest.json"):
        raise ValueError("Bundle/release mismatch")
    split = bundle["split"]
    if split not in {"smoke", "dev", "holdout"}:
        raise ValueError("Invalid split")
    items = {i["question_id"]: i for i in json.loads((release / f"{split}.json").read_text())["items"]}
    candidates = bundle["candidates"]
    if not candidates or any(not isinstance(c, str) or not c for c in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError("Invalid candidates")
    expected = {(c, qid) for c in candidates for qid in items}
    seen = set()
    for case in bundle["cases"]:
        key = (case["candidate"], case["question_id"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate generation case")
        seen.add(key)
        if case["item_sha256"] != fingerprint(items[case["question_id"]]):
            raise ValueError("Generation item hash mismatch")
        if case["status"] not in {"ok", "failed", "authentication_failed", "budget_exceeded", "input_budget_exceeded", "not_submitted"}:
            raise ValueError("Unknown generation status")
        validate_contexts(case["contexts"])
        if case.get("retrieval_contexts") is not None:
            validate_contexts(case["retrieval_contexts"])
        if case["status"] == "ok":
            validate_generation(case["response"])
        elif case["response"] is not None:
            raise ValueError("Failed generation must not masquerade as a scored response")
    if seen != expected:
        raise ValueError("Bundle omits planned requests; use explicit not_submitted rows")
    return bundle, items


def prepare(parent, release, bundle_path, metrics, config, cap):
    if not math.isfinite(cap) or cap <= 0 or not metrics or len(metrics) != len(set(metrics)) or set(metrics) - set(METRICS):
        raise ValueError("Invalid metrics or budget")
    bundle, items = load_bundle(parent, release, bundle_path)
    jobs = []
    for case in bundle["cases"]:
        item = items[case["question_id"]]
        for metric in metrics:
            messages = []
            if metric == "context_precision":
                if item["task_label"] != "answer":
                    continue
                if case.get("retrieval_contexts") is None:
                    continue  # remains explicitly unknown in reporting
                from .context_precision_v11 import messages_for
                messages = messages_for(item, case["retrieval_contexts"])
            else:
                if case["status"] != "ok":
                    continue
                factory = correctness_messages if metric == "correctness" else faithfulness_messages
                messages = [factory(item, case["response"], case["contexts"])]
            task = {"candidate": case["candidate"], "question_id": case["question_id"], "metric": metric,
                    "case_sha256": fingerprint(case), "messages": messages}
            jobs.append({"task_id": fingerprint(task), **task})
    paths = [Path(__file__), Path("src/evaluation/rubric_v11.py"), Path("src/evaluation/prompts_v1.py"),
             Path("src/evaluation/paid_calls_v1.py"), Path("src/evaluation/context_precision_v11.py"),
             Path("src/evaluation/evidence.py"), Path("src/generation/prompt.py")]
    binding = {"runner": "score-v1.1-1", "split": bundle["split"], "origin": bundle["origin"],
        "release_manifest_sha256": file_sha256(release / "release_manifest.json"), "bundle_sha256": file_sha256(bundle_path),
        "metrics": list(metrics), "judge": dataclasses.asdict(config), "budget_cap_rmb": cap,
        "pricing": PRICING, "ragas_version": importlib.metadata.version("ragas") if "context_precision" in metrics else None,
        "code_sha256": {str(p.relative_to(Path.cwd()) if p.is_absolute() else p): file_sha256(p) for p in paths},
        "task_ids": [j["task_id"] for j in jobs]}
    messages = [m for j in jobs for m in j["messages"]]
    unique = {fingerprint({"metric": j["metric"], "messages": m}): m for j in jobs for m in j["messages"]}
    plan = {"binding": binding, "binding_sha256": fingerprint(binding), "tasks": jobs,
        "planned_questions_per_candidate": len(items), "metric_tasks": len(jobs),
        "planned_verdict_calls_before_cache": len(messages), "unique_verdict_inputs": len(unique),
        "max_attempts_before_cache": len(messages) * config.max_attempts,
        "conservative_peak_byte_bound_rmb": sum(price(config.model, input_upper_bound(m), config.max_tokens) * config.max_attempts for m in messages),
        "forecast_note": "Byte/token upper bound, not expected spend; cap limits execution. Context Precision is one verdict per retrieved chunk.",
        "input_limit_exceeding_calls": sum(input_upper_bound(m) > config.input_token_upper_limit for m in messages),
        "independent_calibration_complete": False, "holdout_bundle": bundle["split"] == "holdout"}
    return plan, bundle, items


def score_task(job, case, item, caller, config, role):
    metric = job["metric"]
    if metric == "context_precision":
        from .context_precision_v11 import evaluate
        return evaluate(item, case["retrieval_contexts"], caller, config, role)
    valid_ids = {c["chunk_id"] for c in case["contexts"]}
    validator = validate_correctness if metric == "correctness" else lambda v: score_units(v, case["response"]["answer"], valid_ids)
    call = caller.call(role, job["messages"][0], config, validator)
    result = {"status": call["status"], "calls": [call], "score": None}
    if call["status"] == "ok":
        validator(call["parsed"])
        if metric == "correctness":
            result.update(score=int(call["parsed"]["correct"]), verdict=call["parsed"])
        else:
            result.update(score_units(call["parsed"], case["response"]["answer"], valid_ids), units=call["parsed"]["units"])
    return result


def load_result(out, job, binding_sha):
    path = out / "results" / (job["task_id"] + ".json")
    if not path.exists():
        return None
    value = read_record(path)
    if value["task_id"] != job["task_id"] or value["binding_sha256"] != binding_sha:
        raise ValueError("Judgment belongs to another plan")
    return value["result"]


def summarize(cases, items, results, requested):
    n, correct, unknown, failed = len(cases), 0, 0, 0
    faith, faith_unknown, no_facts = [], 0, 0
    cp, cp_unknown = [], 0
    for case in cases:
        qid = case["question_id"]
        def result(metric):
            value = results.get((case["candidate"], qid, metric))
            return value if value and value["status"] == "ok" else None
        if case["status"] == "not_submitted":
            unknown += 1
        elif case["status"] != "ok":
            failed += 1
        else:
            correctness = result("correctness")
            if correctness is None:
                unknown += 1
            else:
                correct += correctness["score"]
            f = result("faithfulness")
            if f is None:
                faith_unknown += 1
            elif f["score"] is None:
                no_facts += 1
            else:
                faith.append(f["score"])
        if items[qid]["task_label"] == "answer":
            c = result("context_precision")
            if c is None:
                cp_unknown += 1
            else:
                cp.append(c["score"])
    def bounds(values, missing):
        denominator = len(values) + missing
        return [sum(values) / denominator, (sum(values) + missing) / denominator] if denominator else None
    return {"n": n, "generation_status_counts": dict(Counter(c["status"] for c in cases)),
        "correctness": {"requested": "correctness" in requested, "correct": correct, "unknown": unknown, "generation_failures": failed,
                        "bounds": [correct / n, (correct + unknown) / n] if n else None},
        "faithfulness": {"requested": "faithfulness" in requested, "scored_claim_bearing_n": len(faith), "known_no_fact_n": no_facts,
            "unknown_eligibility_or_score_n": faith_unknown, "mean_scored": sum(faith) / len(faith) if faith else None,
            "conservative_bounds_allowing_missing_to_be_fact_bearing": bounds(faith, faith_unknown)},
        "context_precision": {"requested": "context_precision" in requested, "eligible_answer_n": len(cp) + cp_unknown,
            "scored_n": len(cp), "unknown_n": cp_unknown, "bounds": bounds(cp, cp_unknown)}}


def report(plan, bundle, items, out, ledger):
    results = {(j["candidate"], j["question_id"], j["metric"]): load_result(out, j, plan["binding_sha256"]) for j in plan["tasks"]}
    metrics = plan["binding"]["metrics"]
    summaries = {}
    for name in bundle["candidates"]:
        rows = [c for c in bundle["cases"] if c["candidate"] == name]
        summaries[name] = {"overall": summarize(rows, items, results, metrics),
            "by_label": {label: summarize([c for c in rows if items[c["question_id"]]["task_label"] == label], items, results, metrics)
                         for label in ("answer", "clarify", "refuse")}}
    return {"scope": "v1.1 judgments; uncalibrated, no final quality or latency acceptance claim",
        "split": bundle["split"], "origin": bundle["origin"], "binding_sha256": plan["binding_sha256"],
        "summaries": summaries, "completed_metric_tasks": sum(v is not None for v in results.values()),
        "successful_metric_tasks": sum(v is not None and v["status"] == "ok" for v in results.values()),
        "pending_metric_tasks": sum(v is None for v in results.values()),
        "new_api_attempts": len(ledger.receipts()), "accounted_upper_rmb": ledger.spent_upper(),
        "unknown_usage_attempts": sum(r.get("usage_unknown", True) for r in ledger.receipts()),
        "independent_calibration_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("import-legacy", "plan", "judge", "report"))
    parser.add_argument("--parent", type=Path, default=Path("doc/evaluation/v1/releases/v1.0"))
    parser.add_argument("--release", type=Path, default=Path("doc/evaluation/v1/releases/v1.1"))
    parser.add_argument("--legacy-run", type=Path, default=Path("data/experiments/eval_v1/generation_smoke_v1"))
    parser.add_argument("--bundle", type=Path, default=Path("data/experiments/eval_v11/reused_smoke_generations.json"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/eval_v11/scoring_smoke"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["correctness", "faithfulness"])
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--budget-rmb", type=float, default=2.0)
    parser.add_argument("--max-tasks", type=int, help="Execute at most this many pending tasks; keep the full denominator")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.phase == "import-legacy":
        print(json.dumps(import_legacy(args.parent, args.release, args.legacy_run, args.bundle)))
        return
    if args.max_tasks is not None and args.max_tasks <= 0:
        parser.error("--max-tasks must be positive")
    config = CallConfig(model=args.model)
    plan, bundle, items = prepare(args.parent, args.release, args.bundle, args.metrics, config, args.budget_rmb)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.out / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Scoring inputs/code/config changed; use a new output directory")
        elif args.phase != "plan":
            raise ValueError("Create and inspect the offline plan before judging/reporting")
        else:
            atomic_json(plan_path, plan)
        ledger = BudgetLedger(args.out / "usage.jsonl", args.budget_rmb)
        pending = [j for j in plan["tasks"] if load_result(args.out, j, plan["binding_sha256"]) is None]
        print(json.dumps({"tasks": plan["metric_tasks"], "verdict_calls_before_cache": plan["planned_verdict_calls_before_cache"],
            "unique_verdict_inputs": plan["unique_verdict_inputs"], "pending_tasks": len(pending), "budget_cap_rmb": args.budget_rmb,
            "conservative_peak_byte_bound_rmb": plan["conservative_peak_byte_bound_rmb"]}), flush=True)
        if args.phase == "judge" and pending:
            if args.env_file:
                from dotenv import load_dotenv
                load_dotenv(args.env_file, override=False)
            caller = JsonCaller(args.out / "api_cache", ledger)
            by_key = {(c["candidate"], c["question_id"]): c for c in bundle["cases"]}
            for job in pending[:args.max_tasks]:
                case = by_key[(job["candidate"], job["question_id"])]
                role = "v11:" + job["metric"] + ":" + plan["binding_sha256"]
                result = score_task(job, case, items[job["question_id"]], caller, config, role)
                value = {"task_id": job["task_id"], "binding_sha256": plan["binding_sha256"], "result": result}
                atomic_json(args.out / "results" / (job["task_id"] + ".json"), {"payload": value, "sha256": fingerprint(value)})
                print(json.dumps({"metric": job["metric"], "status": result["status"]}), flush=True)
                if result["status"] in {"budget_exceeded", "authentication_failed"}:
                    break
        result = report(plan, bundle, items, args.out, ledger)
        atomic_json(args.out / "report.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
