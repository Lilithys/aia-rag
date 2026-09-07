"""Offline reuse audit and blinded review pack. Never imports an API client.

Keep generation provenance and failures, invalidate old correctness judgments,
and prepare complete new judge payloads without submitting them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.artifacts import file_sha256, fingerprint
from src.data_prep.revise_eval_v11 import AMBIGUOUS, GENERAL, validate_revision
from .prompts_v1 import generation_messages
from .rubric_v11 import correctness_messages, faithfulness_messages, scope_fixtures, score_units


def read_record(path):
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "sha256"} or fingerprint(envelope["payload"]) != envelope["sha256"]:
        raise ValueError(f"Corrupt record: {path}")
    return envelope["payload"]


def check_generation_reuse(before, after, generated, config):
    if generated["question_id"] != before["question_id"] or generated["item_sha256"] != fingerprint(before):
        raise ValueError("Generation belongs to another original item")
    chunks = [SimpleNamespace(**c) for c in generated["contexts"]]
    old_messages, new_messages = generation_messages(before, chunks), generation_messages(after, chunks)
    if old_messages != new_messages:
        raise ValueError("Generator inputs changed; do not reuse this response")
    call = generated.get("call")
    if call:
        expected = fingerprint({"schema_version": 1, "role": "generation", "messages": old_messages, "config": config})
        if call["request_hash"] != expected:
            raise ValueError("Recorded API request differs from reconstructed inputs")
    elif generated.get("gate") != "low_retrieval_confidence":
        raise ValueError("Cannot establish request provenance")
    return fingerprint(new_messages)


def build_pack(parent, release, run):
    validate_revision(parent, release)
    plan = json.loads((run / "plan.json").read_text())
    binding = plan["binding"]
    if fingerprint(binding) != plan["binding_sha256"]:
        raise ValueError("Corrupt parent plan")
    if file_sha256(parent / "smoke.json") != binding["dataset_sha256"] or file_sha256(parent / "release_manifest.json") != binding["release_manifest_sha256"]:
        raise ValueError("Parent run/release mismatch")
    for name, sha in binding["code_sha256"].items():
        if file_sha256(name) != sha:
            raise ValueError(f"Original execution code changed: {name}")
    old = {i["question_id"]: i for i in json.loads((parent / "smoke.json").read_text())["items"]}
    new = {i["question_id"]: i for i in json.loads((release / "smoke.json").read_text())["items"]}
    if list(old) != list(new) or list(old) != binding["question_ids"]:
        raise ValueError("Smoke IDs/order changed")
    rows, review, mappings, pending, examples = [], [], [], [], []
    for name in binding["candidates"]:
        for qid, item in new.items():
            path = run / "generation" / name / (fingerprint(qid) + ".json")
            generated = read_record(path)
            if generated["candidate"] != name:
                raise ValueError("Wrong candidate checkpoint")
            inputs_sha = check_generation_reuse(old[qid], item, generated, binding["generator"])
            blind_id = fingerprint({"salt": "calibration-1.1-blind", "qid": qid, "generation": fingerprint(generated)})[:24]
            successful = generated["status"] == "ok"
            rows.append({"candidate": name, "question_id": qid, "generation_file": str(path),
                "generation_file_sha256": file_sha256(path), "old_item_sha256": fingerprint(old[qid]),
                "new_item_sha256": fingerprint(item), "generator_messages_sha256": inputs_sha,
                "status": generated["status"], "generation_reusable": True,
                "correctness_state": "needs_new_judgment" if successful else "generation_failure_zero",
                "old_latency_retained_not_remeasured": True})
            mappings.append({"blind_id": blind_id, "candidate": name, "question_id": qid, "generation_file": str(path)})
            data = json.loads(correctness_messages(item, generated["response"], generated["contexts"])[1]["content"])
            review.append({"blind_id": blind_id, "question": item["question"], "history": item["conversation_history"],
                "task_label": item["task_label"], "reference": item["gold_answer"],
                "acceptance_criteria": item.get("acceptance_criteria", {}),
                "reference_source_blocks": data["reference_source_blocks"],
                "contexts": generated["contexts"], "response": generated["response"],
                "generation_status": generated["status"],
                "human_correct": None, "human_reason": None, "human_scoped_units": None,
                "reviewer_id": None, "reviewed_at": None})
            if successful:
                for metric, messages in (("correctness", correctness_messages(item, generated["response"], generated["contexts"])),
                                         ("faithfulness_scoped_candidate", faithfulness_messages(item, generated["response"], generated["contexts"]))):
                    pending.append({"blind_id": blind_id, "metric": metric, "messages": messages,
                                    "input_sha256": fingerprint({"release": file_sha256(release / "release_manifest.json"), "metric": metric, "messages": messages})})
            if qid in {AMBIGUOUS, GENERAL}:
                if qid == AMBIGUOUS:
                    note = "Still fails the corrected referent-clarification criterion: the response gives aid/application advice without asking what neither refers to. This is an AI review, not a new judge result."
                elif name == "rerank_truncated_k10":
                    note = "General explanation now addresses the request, but the long restoration checklist needs claim-by-claim review for branch restrictions; do not automatically flip the verdict."
                else:
                    note = "General explanation addresses the corrected request. Review all added facts against the supplied contexts; generic second-person language alone is not a personal-status assertion."
                examples.append({"blind_id": blind_id, "question_id": qid, "old_reference": old[qid]["gold_answer"],
                                 "new_reference": item["gold_answer"], "response": generated["response"],
                                 "agent_review_note": note, "independent_human_review": False})
    fixtures = scope_fixtures()
    fixture_results = [{"case_id": f["case_id"], **score_units(f["expected"], f["answer"], {c["chunk_id"] for c in f["contexts"]})} for f in fixtures]
    review.sort(key=lambda r: r["blind_id"])
    summary = {"status": "ready_for_independent_review_and_budgeted_judge_calibration", "paid_api_calls": 0,
        "additional_cost_rmb": 0, "generation_records_reusable": len(rows),
        "successful_outputs": sum(r["status"] == "ok" for r in rows),
        "generation_failures_preserved": sum(r["status"] != "ok" for r in rows),
        "proposed_metric_inputs": len(pending), "model_judgments_executed": 0,
        "old_correctness_scores_reused": 0, "changed_item_response_reviews": len(examples),
        "scope_fixture_count": len(fixtures), "independent_human_review": False,
        "judge_semantic_calibration_passed": False, "holdout_outputs_used": False,
        "limitations": ["Fixture annotations are AI-authored; scoring them tests the contract, not model scope classification.",
                        "Full-response extraction and semantic entailment still need independent checks.",
                        "Share blind_review.json plus rubric instructions with the reviewer; keep mapping and agent review notes separate.",
                        "Retained latency belongs to the original v1.0 run. No new run or quality gain is claimed.",
                        "Proposed metric inputs have no model/budget authorization attached and this command cannot submit them."]}
    provenance = {"parent_plan_sha256": file_sha256(run / "plan.json"), "revision_manifest_sha256": file_sha256(release / "release_manifest.json"),
        "original_usage_ledger_sha256": file_sha256(run / "usage.jsonl"),
        "code_sha256": {str(p): file_sha256(p) for p in [Path(__file__), Path("src/evaluation/rubric_v11.py")]}}
    return {"summary": summary, "provenance": provenance, "generation_reuse": rows,
            "blind_review": review, "private_mapping": mappings, "pending_judge_inputs": pending,
            "agent_review_notes": examples, "scope_fixtures": fixtures, "fixture_contract_check": fixture_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("doc/evaluation/v1/releases/v1.0"))
    parser.add_argument("--release", type=Path, default=Path("doc/evaluation/v1/releases/v1.1"))
    parser.add_argument("--run", type=Path, default=Path("data/experiments/eval_v1/generation_smoke_v1"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/eval_v11/calibration_pack"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Review pack exists; use a new output path")
    values = build_pack(args.parent, args.release, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-pack-", dir=args.out.parent))
    try:
        for name, value in values.items():
            (staging / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        manifest = {"files": {p.name: file_sha256(p) for p in sorted(staging.iterdir())}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.out.exists():
            raise FileExistsError("Review pack appeared during preparation")
        staging.rename(args.out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(values["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
