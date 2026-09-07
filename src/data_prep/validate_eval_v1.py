"""Verify frozen evaluation files without importing RAG or calling any API."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .freeze_eval_v1 import _checked_file, validate_source_inventory


def validate_release(release_dir: Path, source_root: Path | None = None) -> dict:
    manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    expected_files = {"dev", "holdout", "smoke", "source_inventory", "protocol"}
    if set(manifest["files"]) != expected_files:
        raise ValueError("release file inventory is incomplete or unexpected")
    paths = {key: _checked_file(release_dir, entry) for key, entry in manifest["files"].items()}
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    inventory = json.loads(paths["source_inventory"].read_text(encoding="utf-8"))
    if protocol["protocol_version"] != manifest["release_version"]:
        raise ValueError("release/protocol version mismatch")
    if inventory["corpus_fingerprint_sha256"] != manifest["source_corpus_fingerprint_sha256"]:
        raise ValueError("release/inventory corpus mismatch")
    recorded_eval = next(f for f in inventory["source_files"] if f["path"] == "test/eval_turns.json")
    if recorded_eval["sha256"] != manifest["source_eval_sha256"]:
        raise ValueError("release/inventory eval mismatch")
    splits = {}
    for split in ("dev", "holdout", "smoke"):
        doc = json.loads(paths[split].read_text(encoding="utf-8"))
        items = doc["items"]
        splits[split] = items
        ids = [i["question_id"] for i in items]
        dialogues = [i["source"]["dialogue_id"] for i in items]
        if len(ids) != len(set(ids)) or len(dialogues) != len(set(dialogues)):
            raise ValueError(f"duplicate IDs/dialogues: {split}")
        if ids != manifest["question_ids"][split]:
            raise ValueError(f"ID/order mismatch: {split}")
        if len(items) != doc["total_questions"] or len(items) != manifest["files"][split]["questions"]:
            raise ValueError(f"count mismatch: {split}")
        targets = protocol["smoke_task_target"] if split == "smoke" else protocol["task_target_per_split"]
        actual = Counter(i["task_label"] for i in items)
        if actual != Counter(targets) or dict(actual) != manifest["task_label_distribution_per_split"][split]:
            raise ValueError(f"label quota mismatch: {split}")
        if doc["evaluation_release"] != manifest["release_version"] or doc["split"] != split:
            raise ValueError(f"split/version mismatch: {split}")
    dev_dialogues = {i["source"]["dialogue_id"] for i in splits["dev"]}
    holdout_dialogues = {i["source"]["dialogue_id"] for i in splits["holdout"]}
    if dev_dialogues & holdout_dialogues:
        raise ValueError("dev/holdout dialogue overlap")
    if holdout_dialogues & set(inventory["holdout_excluded_dialogue_ids"]):
        raise ValueError("holdout includes a historically exposed dialogue")
    dev = {i["question_id"]: i for i in splits["dev"]}
    if set(dev) & {i["question_id"] for i in splits["holdout"]}:
        raise ValueError("dev/holdout question overlap")
    for item in splits["smoke"]:
        if dev.get(item["question_id"]) != item:
            raise ValueError("smoke is not an exact subset of dev")
    if manifest["judge_calibration_question_ids"] != manifest["question_ids"]["smoke"]:
        raise ValueError("calibration IDs differ from fixed smoke")
    if source_root is not None:
        validate_source_inventory(inventory, source_root)
    return {
        "release_version": manifest["release_version"],
        "questions": {s: len(items) for s, items in splits.items()},
        "documents": len(inventory["documents"]),
        "source_bytes_verified": source_root is not None,
        "dialogue_overlap": 0,
        "independent_human_review": manifest["review_provenance"]["independent_human_review"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=Path("doc/evaluation/v1/releases/v1.0"))
    parser.add_argument("--source-root", type=Path, help="Also verify all original document and test bytes")
    args = parser.parse_args(argv)
    try:
        result = validate_release(args.release_dir, args.source_root)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Verification failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
