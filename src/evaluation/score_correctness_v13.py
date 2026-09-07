"""CLI for budgeted correctness-v1.3 scoring of a frozen generation bundle."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from dotenv import load_dotenv

from src.artifacts import fingerprint
from src.evaluation import correctness_v13
from src.evaluation.paid_calls_v1 import BudgetLedger, CallConfig, JsonCaller, atomic_json
from src.evaluation.score_v11 import load_result, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "judge", "report"))
    parser.add_argument("--parent", type=Path, default=Path("doc/evaluation/v1/releases/v1.0"))
    parser.add_argument("--release", type=Path, default=Path("doc/evaluation/v1/releases/v1.1"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-rmb", type=float, default=1.5)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    args = parser.parse_args()
    config = CallConfig()
    plan, bundle, items = correctness_v13.prepare(args.parent, args.release, args.bundle, config, args.budget_rmb)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.out / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Scoring inputs/code/config changed; use a new output directory")
        elif args.phase != "plan":
            raise ValueError("Create and inspect the offline plan before judging")
        else:
            atomic_json(plan_path, plan)
        ledger = BudgetLedger(args.out / "usage.jsonl", args.budget_rmb)
        pending = [job for job in plan["tasks"] if load_result(args.out, job, plan["binding_sha256"]) is None]
        print(json.dumps({
            "tasks": len(plan["tasks"]),
            "pending": len(pending),
            "max_attempts": len(plan["tasks"]) * config.max_attempts,
            "model": config.model,
            "thinking": config.thinking,
            "budget_cap_rmb": args.budget_rmb,
            "conservative_peak_byte_bound_rmb": plan["conservative_peak_byte_bound_rmb"],
        }), flush=True)
        if args.phase == "judge" and pending:
            load_dotenv(args.env_file, override=False)
            caller = JsonCaller(args.out / "api_cache", ledger)
            by_key = {(case["candidate"], case["question_id"]): case for case in bundle["cases"]}
            for job in pending:
                case = by_key[(job["candidate"], job["question_id"])]
                item = items[job["question_id"]]
                result = correctness_v13.score_task(
                    job, case, item, caller, config,
                    "final:correctness-v13:" + plan["binding_sha256"],
                )
                value = {"task_id": job["task_id"], "binding_sha256": plan["binding_sha256"], "result": result}
                atomic_json(args.out / "results" / (job["task_id"] + ".json"), {
                    "payload": value, "sha256": fingerprint(value),
                })
                print(json.dumps({"question_id": job["question_id"], "status": result["status"]}), flush=True)
                if result["status"] in {"budget_exceeded", "authentication_failed"}:
                    break
        result = report(plan, bundle, items, args.out, ledger)
        result["scope"] = "Correctness v1.3 automated proxy; diagnostic coverage is limited and no independent human calibration was performed."
        atomic_json(args.out / "report.json", result)
        print(json.dumps({
            "completed": result["completed_metric_tasks"],
            "successful": result["successful_metric_tasks"],
            "pending": result["pending_metric_tasks"],
            "accounted_rmb": result["accounted_upper_rmb"],
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
