"""Validate reviewed evaluation-v1 candidates and publish immutable test files.

The command publishes nothing while a review is pending, rejected, ambiguous,
or any pinned input differs. Review edits belong in the decision ledgers;
regenerate the candidate package before freezing. All validation is offline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


from .build_eval_v1_candidates import DEFAULT_SOURCE_ROOT, _convert_candidate

DEFAULT_CANDIDATES_DIR = Path("doc/evaluation/v1/final_candidates")
DEFAULT_OUT_DIR = Path("doc/evaluation/v1/releases/v1.0")
DEFAULT_INVENTORY = Path("doc/evaluation/v1/source_inventory.json")
DEFAULT_PROTOCOL = Path("doc/evaluation/v1/evaluation_protocol.json")
REVIEW_FIELDS = {
    "review_status",
    "task_label",
    "reviewed_reference",
    "label_reason",
    "reviewer_notes",
}
ALLOWED_LABELS = {"answer", "clarify", "refuse"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_reviews(path: Path, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = (REVIEW_FIELDS | {"question_id"}) - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"{path} missing review fields: {sorted(missing_fields)}")
        rows = list(reader)
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate question_id values")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        raise ValueError(f"{path} question IDs differ from candidate JSON; missing={missing}, extra={extra}")
    return {row["question_id"]: {field: row[field].strip() for field in REVIEW_FIELDS} for row in rows}


def _group_evidence(flat_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for evidence in flat_evidence:
        doc_id = evidence["doc_id"]
        if doc_id not in grouped:
            grouped[doc_id] = {
                "doc_id": doc_id,
                "document_file": evidence.get("document_file"),
                "document_format": evidence.get("document_format"),
                "document_language": evidence.get("document_language"),
                "spans": [],
            }
        grouped[doc_id]["spans"].append(
            {
                key: evidence.get(key)
                for key in (
                    "source_turn",
                    "label",
                    "span_id",
                    "source_span_text_en",
                    "source_block_id",
                    "evidence_text",
                    "location",
                )
                if evidence.get(key) is not None
            }
        )
    return list(grouped.values())


def _runtime_item(candidate: dict[str, Any], review: dict[str, str]) -> dict[str, Any]:
    label = review["task_label"]
    reference = review["reviewed_reference"]
    if reference == "=SOURCE":
        reference = candidate["source_gold_answer"]
    answerable = label in {"answer", "clarify"}
    source = candidate["source"]
    return {
        "question_id": candidate["question_id"],
        "domain": candidate["domain"],
        "question_type": label,
        "task_label": label,
        "synthetic": bool(candidate.get("synthetic", False)),
        "answerable": answerable,
        "source": {
            "dialogue_id": source["dialogue_id"],
            "question_turn_id": source["question_turn_id"],
            "answer_turn_id": source["answer_turn_id"],
            "agent_dialogue_act": source["agent_dialogue_act"],
            "evaluation_category": source["evaluation_category"],
        },
        "question": candidate["question"],
        "source_question_en": candidate["source_question_en"],
        "conversation_history": [
            {"role": turn["role"], "utterance": turn["utterance"]}
            for turn in candidate["conversation_history"]
        ],
        "gold_answer": reference,
        "source_gold_answer": candidate["source_gold_answer"],
        "source_gold_answer_en": candidate["source_gold_answer_en"],
        "question_language": candidate["question_language"],
        "answer_language": candidate["question_language"],
        "language_relation": candidate["language_relation"],
        "document_format": candidate["document_format"],
        "history_bucket": candidate["history_bucket"],
        "answer_evidence_ids": sorted({
            e["source_block_id"] for e in candidate["evidence"]
            if e.get("source_turn") == "answer" and e.get("source_block_id")
        }) if answerable else [],
        "answer_doc_ids": sorted({
            e["doc_id"] for e in candidate["evidence"]
            if e.get("source_turn") == "answer"
        }) if answerable else [],
        "required_doc_ids": candidate["required_doc_ids"] if answerable else [],
        "required_documents": candidate["required_documents"] if answerable else [],
        "candidate_doc_ids": [] if answerable else candidate["required_doc_ids"],
        "candidate_documents": [] if answerable else candidate["required_documents"],
        "evidence": _group_evidence(candidate["evidence"]) if answerable else [],
        "review": {
            "reviewer_type": "ai_agent",
            "independent_human_review": False,
            "label_reason": review["label_reason"],
            "evidence_ids": candidate["review"]["evidence_ids"],
            "reviewer_notes": review["reviewer_notes"] or None,
        },
    }


def _validate_review(
    split: str,
    candidates: list[dict[str, Any]],
    reviews: dict[str, dict[str, str]],
) -> list[str]:
    problems: list[str] = []
    for candidate in candidates:
        qid = candidate["question_id"]
        review = reviews[qid]
        for field, value in review.items():
            source_field = "status" if field == "review_status" else field
            if value != str(candidate["review"].get(source_field) or "").strip():
                problems.append(f"{split}:{qid}: CSV/JSON review mismatch: {field}")
        status = review["review_status"]
        if status != "approved":
            problems.append(f"{split}:{qid}: review_status={status or 'pending'}")
            continue
        if review["task_label"] not in ALLOWED_LABELS:
            problems.append(f"{split}:{qid}: invalid task_label={review['task_label']!r}")
        if not review["reviewed_reference"]:
            problems.append(f"{split}:{qid}: reviewed_reference is required (or use =SOURCE)")
        if not review["label_reason"]:
            problems.append(f"{split}:{qid}: label_reason is required")
    return problems


def _checked_file(root: Path, record: dict[str, Any]) -> Path:
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"file escapes pinned root: {record['path']}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def validate_source_inventory(inventory: dict[str, Any], source_root: Path) -> None:
    documents = inventory["documents"]
    for field in ("doc_id", "file"):
        if len({d[field] for d in documents}) != len(documents):
            raise ValueError(f"duplicate document {field} in inventory")
    if len(documents) != inventory["counts"]["documents"]:
        raise ValueError("inventory document count mismatch")
    for entry in documents:
        _checked_file(source_root, {"path": entry["file"], "sha256": entry["sha256"]})
    for entry in inventory["source_files"]:
        _checked_file(source_root, entry)


def prepare_release(
    candidates_dir: Path, inventory_path: Path, source_root: Path, protocol_path: Path,
) -> tuple[dict, dict, dict, dict]:
    manifest_path = candidates_dir / "composition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "ready_to_freeze":
        raise ValueError("candidate composition is not ready to freeze")
    if _sha256(inventory_path) != manifest["source_inventory_sha256"]:
        raise ValueError("source inventory SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["corpus_fingerprint_sha256"] != manifest["source_corpus_fingerprint_sha256"]:
        raise ValueError("corpus fingerprint mismatch")
    validate_source_inventory(inventory, source_root)
    source_eval = source_root / "test/eval_turns.json"
    if _sha256(source_eval) != manifest["source_eval_sha256"]:
        raise ValueError("source evaluation SHA-256 mismatch")
    source_items = json.loads(source_eval.read_text(encoding="utf-8"))["items"]
    source_by_id = {i["eval_item_id"]: i for i in source_items}
    if len(source_items) != len(source_by_id):
        raise ValueError("duplicate source eval item ID")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["task_target_per_split"] != manifest["task_target_per_split"]:
        raise ValueError("protocol and composition quotas differ")
    known_documents = {d["doc_id"] for d in inventory["documents"]}
    runtime = {}
    for split in ("dev", "holdout"):
        candidate_path = _checked_file(candidates_dir, manifest["candidate_files"][split])
        review_path = _checked_file(candidates_dir, manifest["review_files"][split])
        document = json.loads(candidate_path.read_text(encoding="utf-8"))
        for key in ("source_eval_sha256", "source_corpus_fingerprint_sha256", "seed", "composer_version", "review_provenance"):
            if document[key] != manifest[key]:
                raise ValueError(f"{split} candidate metadata differs: {key}")
        candidates = document["items"]
        ids = [c["question_id"] for c in candidates]
        dialogues = [c["source"]["dialogue_id"] for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split} has duplicate question IDs")
        if len(dialogues) != len(set(dialogues)):
            raise ValueError(f"{split} has duplicate dialogue IDs")
        if ids != manifest["question_ids"][split]:
            raise ValueError(f"{split} IDs/order differ from manifest")
        reviews = _read_reviews(review_path, set(ids))
        problems = _validate_review(split, candidates, reviews)
        if problems:
            raise ValueError("Review incomplete: " + "; ".join(problems[:5]))
        for candidate in candidates:
            qid = candidate["question_id"]
            if candidate["split"] != split:
                raise ValueError(f"candidate split mismatch: {qid}")
            if not candidate.get("synthetic"):
                # Reviewed references may differ; original inputs/evidence may not.
                original = _convert_candidate(source_by_id[qid], split, candidate["sequence"])
                for key, value in original.items():
                    if key != "review" and candidate[key] != value:
                        raise ValueError(f"source content changed: {qid}: {key}")
            elif reviews[qid]["task_label"] != "refuse" or candidate["required_doc_ids"] or candidate["evidence"]:
                raise ValueError(f"invalid synthetic capability boundary: {qid}")
            if not set(candidate["required_doc_ids"]) <= known_documents:
                raise ValueError(f"unknown required document: {qid}")
            if any(e["doc_id"] not in known_documents for e in candidate["evidence"]):
                raise ValueError(f"unknown evidence document: {qid}")
        runtime[split] = [_runtime_item(c, reviews[c["question_id"]]) for c in candidates]
        if any(not i["gold_answer"].strip() for i in runtime[split]):
            raise ValueError(f"empty resolved reference in {split}")
        if Counter(i["task_label"] for i in runtime[split]) != Counter(protocol["task_target_per_split"]):
            raise ValueError(f"{split} task-label distribution changed")
        for i in runtime[split]:
            if i["task_label"] == "answer" and not i["answer_evidence_ids"]:
                raise ValueError(f"answer has no answer-side evidence: {i['question_id']}")
    dev_dialogues = {i["source"]["dialogue_id"] for i in runtime["dev"]}
    holdout_dialogues = {i["source"]["dialogue_id"] for i in runtime["holdout"]}
    if dev_dialogues & holdout_dialogues:
        raise ValueError("dev/holdout dialogue overlap")
    if holdout_dialogues & set(inventory["holdout_excluded_dialogue_ids"]):
        raise ValueError("holdout includes a historically exposed dialogue")
    if {i["question_id"] for i in runtime["dev"]} & {i["question_id"] for i in runtime["holdout"]}:
        raise ValueError("dev/holdout question ID overlap")
    return runtime, manifest, inventory, protocol


def select_smoke(items: list[dict], targets: dict[str, int], seed: int) -> list[dict]:
    """Quota sampling with soft coverage, using metadata only (no model scores)."""
    selected = []
    dimensions = ("domain", "question_language", "document_format", "history_bucket", "language_relation")
    counts = {d: Counter() for d in dimensions}
    for label in ("refuse", "clarify", "answer"):
        pool = [i for i in items if i["task_label"] == label]
        for slot in range(targets[label]):
            def priority(item: dict) -> tuple:
                coverage = sum(1 / (1 + counts[d][item[d]]) for d in dimensions)
                tie = hashlib.sha256(f"{seed}:smoke:{item['question_id']}".encode()).hexdigest()
                return coverage, tie
            eligible = pool
            if label == "refuse" and targets[label] >= 2:
                # Exercise both domain boundaries and inaccessible personal/live
                # facts in routine smoke/calibration when both are available.
                if slot == 0:
                    eligible = [i for i in pool if i["domain"] == "out_of_domain"] or pool
                elif slot == 1:
                    eligible = [i for i in pool if i["domain"] != "out_of_domain"] or pool
            chosen = max(eligible, key=priority)
            pool.remove(chosen)
            selected.append(chosen)
            for dimension in dimensions:
                counts[dimension][chosen[dimension]] += 1
    selected_ids = {i["question_id"] for i in selected}
    return [i for i in items if i["question_id"] in selected_ids]


def publish_release(args: argparse.Namespace) -> dict:
    if args.out_dir.exists():
        raise FileExistsError(f"release already exists; use a new version: {args.out_dir}")
    runtime, manifest, inventory, protocol = prepare_release(
        args.candidates_dir, args.inventory, args.source_root, args.protocol,
    )
    if args.release_version != protocol["protocol_version"]:
        raise ValueError("release version must match the explicitly versioned protocol")
    runtime["smoke"] = select_smoke(runtime["dev"], protocol["smoke_task_target"], manifest["seed"])
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".eval-release-", dir=args.out_dir.parent))
    try:
        files = {}
        for split, items in runtime.items():
            path = staging / f"{split}.json"
            _write_json(path, {
                "dataset": "MultiDoc2Dial full multilingual evaluation",
                "evaluation_release": args.release_version,
                "protocol_version": protocol["protocol_version"],
                "split": split,
                "total_questions": len(items),
                "review_provenance": manifest["review_provenance"],
                "task_label_distribution": dict(sorted(Counter(i["task_label"] for i in items).items())),
                "items": items,
            })
            files[split] = {"path": path.name, "sha256": _sha256(path), "questions": len(items)}
        for key, source in (("source_inventory", args.inventory), ("protocol", args.protocol)):
            path = staging / f"{key}.json"
            shutil.copyfile(source, path)
            files[key] = {"path": path.name, "sha256": _sha256(path)}
        release_manifest = {
            "release_version": args.release_version,
            "status": "frozen_agent_reviewed_baseline",
            "review_provenance": manifest["review_provenance"],
            "source_candidate_release": manifest["candidate_release"],
            "source_eval_sha256": manifest["source_eval_sha256"],
            "source_corpus_fingerprint_sha256": manifest["source_corpus_fingerprint_sha256"],
            "selection_version": manifest["composer_version"],
            "seed": manifest["seed"],
            "smoke_selection_version": "quota-boundary-soft-coverage-v1",
            "dialogue_overlap": 0,
            "holdout_historically_exposed_dialogue_overlap": 0,
            "split_policy": manifest["split_policy"],
            "review_ledger": manifest["review_ledger"],
            "question_ids": {s: [i["question_id"] for i in items] for s, items in runtime.items()},
            "judge_calibration_question_ids": [i["question_id"] for i in runtime["smoke"]],
            "task_label_distribution_per_split": {
                s: dict(sorted(Counter(i["task_label"] for i in items).items())) for s, items in runtime.items()
            },
            "synthetic_refusal_items_per_split": {
                s: sum(i["synthetic"] for i in items) for s, items in runtime.items()
            },
            "actual_distributions": {
                s: {field: dict(sorted(Counter(i[field] for i in items).items()))
                    for field in ("domain", "question_language", "document_format", "history_bucket", "language_relation")}
                for s, items in runtime.items()
            },
            "candidate_manifest": {"path": "composition_manifest.json", "sha256": _sha256(args.candidates_dir / "composition_manifest.json")},
            "candidate_inputs": manifest["candidate_files"],
            "review_inputs": manifest["review_files"],
            "files": files,
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        if args.out_dir.exists():
            raise FileExistsError(f"release appeared during validation: {args.out_dir}")
        staging.rename(args.out_dir)
        return release_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--release-version", default="1.0")
    args = parser.parse_args(argv)
    try:
        publish_release(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Release blocked: {exc}")
        return 2
    print(f"Published evaluation release {args.release_version} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
