"""Publish/verify the explicit two-item v1.1 correction, without model calls.

This is a derived release, not a resample. Its extra revision manifest makes
the v1.0-only runner reject it rather than silently ignoring revised rubrics.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from src.artifacts import file_sha256, fingerprint
from .validate_eval_v1 import validate_release

PARENT_SHA = "168ef6564c67be368fe83fa2affb864f70045c51ae27a809974690da525ec736"
AMBIGUOUS = "58f62ac630b5ec13054947b7b8520476::agent-turn-10"
GENERAL = "a5e33a8ee5418d4e04d6a9fe980b80f4::agent-turn-2"
PROVENANCE = {
    "reviewer_type": "ai_agent", "independent_human_review": False,
    "evaluated_system_outputs_used_for_selection": False,
    "evaluated_dev_outputs_seen_before_correction": True,
    "selection_note": "No IDs replaced or reordered. Two dev annotations corrected after error analysis; not an independent test.",
}


def corrected_item(original):
    item = copy.deepcopy(original)
    qid = item["question_id"]
    if qid not in {AMBIGUOUS, GENERAL}:
        return item
    item["review"] = dict(PROVENANCE)
    item["review"]["previous_review"] = original["review"]
    if qid == AMBIGUOUS:
        item["gold_answer"] = "What are the two things you mean by 'neither of those'?"
        reason = "The supplied history never names the two credentials; turns 6–8 say hi. Source annotations cannot supply missing conversational facts."
        item["acceptance_criteria"] = {
            "required": ["Ask the user to identify what 'neither of those' refers to before giving a personalized next step."],
            "allowed": ["Briefly explain that the preceding messages do not identify the two things."],
            "reject": ["Assume the user lacks a diploma/GED or needs the homeschooling route.", "Give generic aid/application advice without resolving the referent."],
        }
        keys = ("answer_evidence_ids", "answer_doc_ids", "required_doc_ids", "required_documents", "evidence")
        item["source_annotations_before_correction"] = {k: copy.deepcopy(item[k]) for k in keys}
        for key in keys:
            item[key] = []
        item["retrieval_evaluation"] = {"eligible": False, "reason": "Referent clarification needs conversation repair, not the historical education block."}
        item["review"]["evidence_ids"] = []
    else:
        item["task_label"] = item["question_type"] = "answer"
        item["gold_answer"] = (
            "Revocation means a driver license or driving privilege is canceled. "
            "The person must reapply after the revocation period ends; in most cases DMV approval is needed first. "
            "Tests and fees may also apply. This is a general explanation, not a determination of your personal status."
        )
        reason = "With no prior history, the user requests general details of revocation. Explaining the documented meaning and process does not require assuming that this user received an order."
        item["acceptance_criteria"] = {
            "required": ["Explain cancellation and reapplication after the revocation period in general or conditional terms."],
            "allowed": ["Mention further approval, tests or fees with the source's qualifications.", "Offer a follow-up question after answering the general request."],
            "reject": ["Assert the user's license is revoked or determine personal eligibility.", "Present branch-specific restoration requirements as universal.", "Only ask whether the user received an order without explaining revocation."],
        }
        item["retrieval_evaluation"] = {"eligible": True, "reason": "Existing answer-side block b0016 contains the general explanation."}
        item["review"]["evidence_ids"] = item["answer_evidence_ids"]
    item["review"]["label_reason"] = reason
    return item


def build_payloads(parent: Path):
    if file_sha256(parent / "release_manifest.json") != PARENT_SHA:
        raise ValueError("Unexpected parent manifest; this correction only applies to pinned v1.0")
    validate_release(parent)
    old_manifest = json.loads((parent / "release_manifest.json").read_text())
    payloads, deltas = {}, []
    for split in ("dev", "holdout", "smoke"):
        doc = json.loads((parent / f"{split}.json").read_text())
        originals = doc["items"]
        doc["items"] = [corrected_item(i) for i in originals] if split != "holdout" else copy.deepcopy(originals)
        for before, after in zip(originals, doc["items"]):
            if before != after and split == "dev":
                deltas.append({"question_id": before["question_id"], "before_sha256": fingerprint(before),
                    "after_sha256": fingerprint(after), "changed_fields": sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)),
                    "reason": after["review"]["label_reason"]})
        doc.update(evaluation_release="1.1", protocol_version="1.1", review_provenance=PROVENANCE,
                   task_label_distribution=dict(sorted(Counter(i["task_label"] for i in doc["items"]).items())))
        payloads[split] = doc
    if {d["question_id"] for d in deltas} != {AMBIGUOUS, GENERAL}:
        raise ValueError("Expected both corrections in dev")
    protocol = json.loads((parent / "protocol.json").read_text())
    protocol.update(protocol_version="1.1", status="agent_reviewed_corrections_judge_calibration_pending",
                    runtime_contract="evaluation-revision-1.1", review_provenance=PROVENANCE)
    protocol.pop("task_target_per_split")
    protocol["task_target_by_split"] = {s: d["task_label_distribution"] for s, d in payloads.items()}
    protocol["smoke_task_target"] = payloads["smoke"]["task_label_distribution"]
    protocol["revision_policy"] = {
        "fixed_question_ids_and_histories": True, "holdout_item_payloads_unchanged": True,
        "quota_policy": "Retain the sample. Report 71/19/10 dev and 15/3/2 smoke after relabeling; do not replace examples to restore old quotas.",
        "acceptance_criteria": "Per-item criteria supplement the global rubric; gold is illustrative. Judge only supplied history, never hidden source annotations.",
        "generation_reuse": "Only byte-equivalent generator messages and identical retrieval/generation settings permit reuse; preserve failures, latency and original run provenance.",
        "judge_reuse": "No v1.0 correctness score is a v1.1 result. New protocol/rubric/reference fingerprints require new judgments on all reused outputs.",
        "comparison": "Rescoring effects are evaluation changes, not model improvements. Report paired comparisons within one release/rubric only.",
    }
    protocol["metrics"]["ragas_context_precision"]["denominator"] = "All source answer targets in each split: dev 71, holdout 70; clarify separately, authored refusal excluded."
    protocol["metrics"]["local_retrieval"]["ineligible"] = "The ambiguous-referent clarification has no valid gold retrieval target; report it as not applicable, never as a hit or miss."
    protocol["metrics"]["faithfulness"]["scope"] = "External facts only; context-availability observations remain separately auditable for correctness. Excluding them from faithfulness does not certify their truth. Unsupported external negatives and personal/live assertions remain eligible."
    payloads["protocol"] = protocol
    payloads["source_inventory"] = json.loads((parent / "source_inventory.json").read_text())
    payloads["revision"] = {
        "revision": "evaluation-1.1", "parent_manifest_sha256": PARENT_SHA,
        "review_provenance": PROVENANCE, "changes": deltas,
        "unchanged_holdout_items_sha256": fingerprint(payloads["holdout"]["items"]),
        "unchanged_question_ids": old_manifest["question_ids"],
        "source_corpus_fingerprint_sha256": old_manifest["source_corpus_fingerprint_sha256"],
        "old_results_preserved": True, "new_quality_scores_available": False,
    }
    return payloads


def manifest_for(parent, folder, payloads):
    return {
        "release_version": "1.1", "status": "frozen_agent_reviewed_correction_uncalibrated",
        "parent_manifest_sha256": PARENT_SHA, "review_provenance": PROVENANCE,
        "builder_sha256": file_sha256(Path(__file__)),
        "source_corpus_fingerprint_sha256": payloads["revision"]["source_corpus_fingerprint_sha256"],
        "question_ids": payloads["revision"]["unchanged_question_ids"],
        "task_label_distribution_per_split": payloads["protocol"]["task_target_by_split"],
        "files": {k: {"path": f"{k}.json", "sha256": file_sha256(folder / f"{k}.json")} for k in payloads},
    }


def validate_revision(parent, target):
    expected = build_payloads(parent)
    manifest = json.loads((target / "release_manifest.json").read_text())
    if set(p.name for p in target.iterdir()) != {f"{k}.json" for k in expected} | {"release_manifest.json"}:
        raise ValueError("Unexpected revision files")
    for key, value in expected.items():
        if json.loads((target / f"{key}.json").read_text()) != value:
            raise ValueError(f"Unapproved correction or payload change: {key}")
    if manifest != manifest_for(parent, target, expected):
        raise ValueError("Revision manifest/hash/builder mismatch")
    return {"release_version": "1.1", "documents": len(expected["source_inventory"]["documents"]),
            "questions": {s: len(expected[s]["items"]) for s in ("dev", "holdout", "smoke")},
            "changed_dev_items": 2, "holdout_payloads_unchanged": True, "independent_human_review": False}


def publish(parent, target):
    if target.exists():
        raise FileExistsError("Release exists; verify it or publish a new version")
    payloads = build_payloads(parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".revision-", dir=target.parent))
    try:
        for key, value in payloads.items():
            (staging / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        manifest = manifest_for(parent, staging, payloads)
        (staging / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        validate_revision(parent, staging)
        if target.exists():
            raise FileExistsError("Release appeared during validation")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_revision(parent, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("doc/evaluation/v1/releases/v1.0"))
    parser.add_argument("--target", type=Path, default=Path("doc/evaluation/v1/releases/v1.1"))
    parser.add_argument("--publish", action="store_true", help="Default verifies an existing release")
    args = parser.parse_args()
    print(json.dumps(publish(args.parent, args.target) if args.publish else validate_revision(args.parent, args.target), indent=2))


if __name__ == "__main__":
    main()
