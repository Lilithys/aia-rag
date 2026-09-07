"""One-configuration, resumable holdout generation through the final runtime."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from dotenv import load_dotenv

from src.artifacts import file_sha256, fingerprint
from src.data_prep.revise_eval_v11 import validate_revision
from src.evaluation.paid_calls_v1 import PRICING, atomic_json
from src.evaluation.prepare_calibration_v11 import read_record
from src.evaluation.score_v11 import load_bundle
from src.service.final_runtime import make_runtime, preflight
from src.service.runtime import ServiceConfig


PARENT = Path("doc/evaluation/v1/releases/v1.0")
RELEASE = Path("doc/evaluation/v1/releases/v1.1")


def record_path(root: Path, item: dict) -> Path:
    return root / "generation" / (fingerprint(item["question_id"]) + ".json")


def load_record(root: Path, item: dict, plan: dict):
    path = record_path(root, item)
    if not path.exists():
        return None
    value = read_record(path)
    if (value["binding_sha256"] != plan["binding_sha256"]
            or value["question_id"] != item["question_id"]
            or value["item_sha256"] != fingerprint(item)):
        raise ValueError("Holdout checkpoint input mismatch")
    return value["result"]


def prepare(folder: Path, config: ServiceConfig, budget_rmb: float, split: str):
    if split not in {"smoke", "dev", "holdout"}:
        raise ValueError("Unsupported evaluation split")
    validate_revision(PARENT, RELEASE)
    split_path = RELEASE / f"{split}.json"
    items = json.loads(split_path.read_text())["items"]
    binding = {
        "study": f"final-{split}-generation-v1",
        "split": split,
        "release_manifest_sha256": file_sha256(RELEASE / "release_manifest.json"),
        "split_sha256": file_sha256(split_path),
        "question_ids": [item["question_id"] for item in items],
        "runtime": preflight(folder, config),
        "generation_budget_cap_rmb": budget_rmb,
        "model": config.model,
        "max_attempts_per_question": config.call_config().max_attempts,
        "pricing": PRICING,
        "history_mode": "recorded full history; no closed-loop rollout",
        "termination": "Stop on budget/authentication status; retain every failure and all 100 denominator rows.",
        "holdout_policy": ("One final configuration; no label, question, prompt, index, or parameter changes after exposure."
                           if split == "holdout" else "Development smoke; cannot be reported as final holdout acceptance."),
        "code_sha256": {str(path): file_sha256(path) for path in (
            Path(__file__), Path("src/service/final_runtime.py"), Path("src/service/runtime.py"),
            Path("src/retrieval/folder_retriever.py"), Path("src/evaluation/paid_calls_v1.py"),
            Path("src/evaluation/prompts_v1.py"),
        )},
    }
    return {
        "binding": binding,
        "binding_sha256": fingerprint(binding),
        "planned_questions": len(items),
        "planned_generation_calls": len(items),
        "max_provider_attempts": len(items) * config.call_config().max_attempts,
        "judge_calls": 0,
        "generation_started": False,
    }, items


def export(root: Path, items: list[dict], plan: dict) -> dict:
    candidate = plan["binding"]["runtime"]["config"]["candidate"]
    cases = []
    for item in items:
        result = load_record(root, item, plan)
        cases.append({
            "candidate": candidate,
            "question_id": item["question_id"],
            "item_sha256": fingerprint(item),
            "status": result["status"] if result else "not_submitted",
            "response": result["response"] if result else None,
            "contexts": result["contexts"] if result else [],
            "retrieval_contexts": result["retrieval_contexts"] if result else None,
            "latency_ms": result["latency_ms"] if result else None,
            "generation_cache_hit": result["generation_cache_hit"] if result else None,
            "source_generation_file_sha256": file_sha256(record_path(root, item)) if result else None,
        })
    bundle = {
        "format": "generation-bundle-v1.1",
        "split": plan["binding"]["split"],
        "release_manifest_sha256": plan["binding"]["release_manifest_sha256"],
        "candidates": [candidate],
        "generation_config": plan["binding"],
        "origin": f"frozen final runtime; one {plan['binding']['split']} execution with recorded history",
        "cases": cases,
    }
    atomic_json(root / "generation_bundle.json", bundle)
    summary = {
        "planned": len(cases),
        "submitted": sum(case["status"] != "not_submitted" for case in cases),
        "successful": sum(case["status"] == "ok" for case in cases),
        "failed": sum(case["status"] not in {"ok", "not_submitted"} for case in cases),
    }
    atomic_json(root / "generation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "generate", "report"))
    parser.add_argument("--artifact-folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "dev", "holdout"), default="holdout")
    parser.add_argument("--candidate", choices=("baseline_dense_k10", "rerank_truncated_k10", "rerank_windowed_k5"), default="rerank_windowed_k5")
    parser.add_argument("--budget-rmb", type=float, default=1.5)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    args = parser.parse_args()
    config = ServiceConfig(candidate=args.candidate, temperature=0.0)
    plan, items = prepare(args.artifact_folder, config, args.budget_rmb, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.out / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Holdout inputs or final configuration changed")
        elif args.phase != "plan":
            raise ValueError("Create and inspect the offline holdout plan first")
        else:
            atomic_json(plan_path, plan)
        if args.phase == "generate":
            if args.split == "holdout":
                atomic_json(args.out / "holdout_execution_started.json", {
                    "binding_sha256": plan["binding_sha256"], "used_for_execution": True,
                })
            load_dotenv(args.env_file, override=False)
            pending = [item for item in items if load_record(args.out, item, plan) is None]
            if pending:
                runtime, _ledger = make_runtime(args.artifact_folder, args.out / "runtime", config, args.budget_rmb)
                for item in pending:
                    result = runtime.answer(item["question"], item["conversation_history"],
                                            item["source"]["dialogue_id"], item["source"]["question_turn_id"])
                    value = {"binding_sha256": plan["binding_sha256"], "question_id": item["question_id"],
                             "item_sha256": fingerprint(item), "result": result}
                    atomic_json(record_path(args.out, item), {"payload": value, "sha256": fingerprint(value)})
                    print(json.dumps({"question_id": item["question_id"], "status": result["status"]}), flush=True)
                    if result["status"] in {"budget_exceeded", "authentication_failed"}:
                        break
        summary = export(args.out, items, plan)
        load_bundle(PARENT, RELEASE, args.out / "generation_bundle.json")
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
