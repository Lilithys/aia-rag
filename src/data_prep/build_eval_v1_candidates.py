"""Build deterministic, dialogue-isolated evaluation-v1 review candidates.

This script is intentionally offline: it reads the frozen full-corpus source
and the checked-in source inventory, then writes review candidates. It never
calls a generator, retriever, embedding model, or judge.

The development pool is limited to dialogue IDs already exposed by legacy
tests or smoke runs. The holdout pool excludes every exposed dialogue ID.
One target turn is selected per dialogue so adjacent turns cannot leak across
splits. Source labels are retained separately from task labels. This initial
package is an audit archive; the composer and freezer publish the reviewed
release with explicit reviewer provenance (agent review is not human review).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SAMPLER_VERSION = "dialogue-isolated-greedy-v1"
DEFAULT_SOURCE_ROOT = Path("data/corpus")
DEFAULT_INVENTORY = Path("doc/evaluation/v1/source_inventory.json")
DEFAULT_OUT_DIR = Path("doc/evaluation/v1/candidates")
DEFAULT_REVIEW_OVERRIDES = Path("doc/evaluation/v1/ai_review_overrides.json")

SPLIT_SIZE = 100
CATEGORY_TARGET = {"answer": 70, "clarification": 20, "unanswerable": 10}
DOMAIN_TARGET = {"dmv": 25, "ssa": 25, "studentaid": 25, "va": 25}
LANGUAGE_TARGET = {"en": 50, "zh": 30, "mixed": 20}
RELATION_TARGET = {"aligned": 35, "cross_lingual": 30, "mixed_language": 35}
HISTORY_TARGET = {"none": 15, "short": 30, "medium": 30, "long": 25}
FORMAT_TARGET = {"markdown": 76, "docx": 8, "txt": 6, "pdf_text": 6, "pdf_scanned": 4}
SMOKE_CATEGORY_TARGET = {"answer": 14, "clarification": 4, "unanswerable": 2}

_SUSPECT_TEXT = re.compile(r"(?:\bX{3,}\b|\bH\s+w\s+p\s+p\b|�)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_jitter(seed: int, split: str, item_id: str) -> float:
    raw = hashlib.sha256(f"{seed}:{split}:{item_id}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / (2**64)


def _history_bucket(item: dict[str, Any]) -> str:
    turns = len(item.get("conversation_history", []))
    if turns == 0:
        return "none"
    if turns <= 4:
        return "short"
    if turns <= 8:
        return "medium"
    return "long"


def _document_format(item: dict[str, Any]) -> str:
    documents = item.get("required_documents", [])
    return documents[0].get("format", "unknown") if documents else "unknown"


def _dimension_score(value: str, counts: Counter, targets: dict[str, int]) -> float:
    target = targets.get(value, 0)
    if target <= 0:
        return 0.0
    return (target - counts[value]) / target


def _pick_split(
    pool: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
    category_target: dict[str, int],
) -> list[dict[str, Any]]:
    """Greedily meet exact category quotas while balancing other dimensions."""
    used_dialogues: set[str] = set()
    selected: list[dict[str, Any]] = []
    counts = {
        "domain": Counter(),
        "language": Counter(),
        "relation": Counter(),
        "history": Counter(),
        "format": Counter(),
    }

    # Rarest/most label-sensitive categories go first so common answer turns
    # cannot consume dialogues needed for refusal and clarification coverage.
    for category in ("unanswerable", "clarification", "answer"):
        for _ in range(category_target[category]):
            candidates = [
                item
                for item in pool
                if item["evaluation_category"] == category
                and item["dialogue_id"] not in used_dialogues
            ]
            if not candidates:
                raise ValueError(f"cannot satisfy {split} category target {category_target}")

            def score(item: dict[str, Any]) -> tuple[float, float]:
                balance = (
                    3.0 * _dimension_score(item["domain"], counts["domain"], DOMAIN_TARGET)
                    + 2.0
                    * _dimension_score(
                        item["dialogue_language"], counts["language"], LANGUAGE_TARGET
                    )
                    + 1.0
                    * _dimension_score(
                        item["language_relation"], counts["relation"], RELATION_TARGET
                    )
                    + 1.0
                    * _dimension_score(
                        _history_bucket(item), counts["history"], HISTORY_TARGET
                    )
                    + 1.0
                    * _dimension_score(
                        _document_format(item), counts["format"], FORMAT_TARGET
                    )
                )
                return balance, _stable_jitter(seed, split, item["eval_item_id"])

            chosen = max(candidates, key=score)
            selected.append(chosen)
            used_dialogues.add(chosen["dialogue_id"])
            counts["domain"][chosen["domain"]] += 1
            counts["language"][chosen["dialogue_language"]] += 1
            counts["relation"][chosen["language_relation"]] += 1
            counts["history"][_history_bucket(chosen)] += 1
            counts["format"][_document_format(chosen)] += 1

    # Keep file order deterministic but unrelated to category selection order.
    rng = random.Random(f"{seed}:{split}:order")
    rng.shuffle(selected)
    return selected


def _quality_flags(item: dict[str, Any]) -> list[str]:
    flags = []
    question = item.get("question", "")
    reference = item.get("gold_answer", "")
    if item["evaluation_category"] in {"clarification", "unanswerable"}:
        flags.append("source_label_requires_review")
    if item.get("dialogue_language") != "en":
        flags.append("translated_text_review")
    if _SUSPECT_TEXT.search(question) or _SUSPECT_TEXT.search(reference):
        flags.append("suspect_placeholder_or_garbled_text")
    if len(question.strip()) < 8:
        flags.append("very_short_question")
    if len(reference.strip()) < 12:
        flags.append("very_short_reference")
    if item["evaluation_category"] in {"answer", "clarification"} and not any(
        evidence.get("source_turn") == "answer" for evidence in item.get("evidence", [])
    ):
        flags.append("missing_answer_evidence")
    return flags


def _convert_candidate(item: dict[str, Any], split: str, sequence: int) -> dict[str, Any]:
    source_category = item["evaluation_category"]
    evidence_ids = sorted(
        {
            evidence["source_block_id"]
            for evidence in item.get("evidence", [])
            if evidence.get("source_turn") == "answer" and evidence.get("source_block_id")
        }
    )
    return {
        "candidate_schema_version": 1,
        "split": split,
        "sequence": sequence,
        "question_id": item["eval_item_id"],
        "source": {
            "dialogue_id": item["dialogue_id"],
            "question_turn_id": item["question_turn_id"],
            "answer_turn_id": item["answer_turn_id"],
            "agent_dialogue_act": item["agent_dialogue_act"],
            "evaluation_category": source_category,
        },
        "domain": item["domain"],
        "question_language": item["dialogue_language"],
        "language_relation": item["language_relation"],
        "document_format": _document_format(item),
        "history_bucket": _history_bucket(item),
        "question": item["question"],
        "source_question_en": item.get("source_question_en"),
        "conversation_history": item.get("conversation_history", []),
        "source_gold_answer": item["gold_answer"],
        "source_gold_answer_en": item.get("source_gold_answer_en"),
        "required_doc_ids": item.get("required_doc_ids", []),
        "gold_doc_ids": item.get("gold_doc_ids", []),
        "required_documents": item.get("required_documents", []),
        "evidence": item.get("evidence", []),
        "review": {
            "status": "pending",
            "task_label": None,
            "reviewed_reference": None,
            "label_reason": None,
            "evidence_ids": evidence_ids,
            "reviewer_notes": None,
            "quality_flags": _quality_flags(item),
        },
    }


def _distribution(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = {
        "source_category": lambda item: item["source"]["evaluation_category"],
        "domain": lambda item: item["domain"],
        "question_language": lambda item: item["question_language"],
        "language_relation": lambda item: item["language_relation"],
        "document_format": lambda item: item["document_format"],
        "history_bucket": lambda item: item["history_bucket"],
    }
    return {
        name: dict(sorted(Counter(getter(item) for item in items).items()))
        for name, getter in fields.items()
    }


def _apply_review_overrides(
    items_by_split: dict[str, list[dict[str, Any]]], overrides_path: Path
) -> str | None:
    if not overrides_path.exists():
        return None
    document = json.loads(overrides_path.read_text(encoding="utf-8"))
    lookup = {
        (item["split"], item["question_id"]): item
        for items in items_by_split.values()
        for item in items
    }
    for override in document["items"]:
        key = (override["split"], override["question_id"])
        if key not in lookup:
            raise ValueError(f"manual review override does not match this candidate release: {key}")
        lookup[key]["review"].update(
            {
                field: override.get(field)
                for field in (
                    "status",
                    "task_label",
                    "reviewed_reference",
                    "label_reason",
                    "reviewer_notes",
                )
                if field in override
            }
        )
    return _sha256(overrides_path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_review_csv(path: Path, items: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "sequence",
        "question_id",
        "dialogue_id",
        "domain",
        "question_language",
        "language_relation",
        "source_category",
        "document_format",
        "history_bucket",
        "history_turns",
        "conversation_history_json",
        "question",
        "source_question_en",
        "source_gold_answer",
        "source_gold_answer_en",
        "required_doc_ids",
        "answer_evidence_ids",
        "answer_evidence_text",
        "quality_flags",
        "review_status",
        "task_label",
        "reviewed_reference",
        "label_reason",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "split": item["split"],
                    "sequence": item["sequence"],
                    "question_id": item["question_id"],
                    "dialogue_id": item["source"]["dialogue_id"],
                    "domain": item["domain"],
                    "question_language": item["question_language"],
                    "language_relation": item["language_relation"],
                    "source_category": item["source"]["evaluation_category"],
                    "document_format": item["document_format"],
                    "history_bucket": item["history_bucket"],
                    "history_turns": len(item["conversation_history"]),
                    "conversation_history_json": json.dumps(
                        item["conversation_history"], ensure_ascii=False
                    ),
                    "question": item["question"],
                    "source_question_en": item["source_question_en"],
                    "source_gold_answer": item["source_gold_answer"],
                    "source_gold_answer_en": item["source_gold_answer_en"],
                    "required_doc_ids": json.dumps(item["required_doc_ids"], ensure_ascii=False),
                    "answer_evidence_ids": json.dumps(item["review"]["evidence_ids"], ensure_ascii=False),
                    "answer_evidence_text": json.dumps(
                        [
                            evidence.get("evidence_text")
                            for evidence in item["evidence"]
                            if evidence.get("source_turn") == "answer"
                        ],
                        ensure_ascii=False,
                    ),
                    "quality_flags": ";".join(item["review"]["quality_flags"]),
                    "review_status": item["review"]["status"],
                    "task_label": item["review"]["task_label"] or "",
                    "reviewed_reference": item["review"]["reviewed_reference"] or "",
                    "label_reason": item["review"]["label_reason"] or "",
                    "reviewer_notes": item["review"]["reviewer_notes"] or "",
                }
            )


def _validate(
    dev: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    exposed_ids: set[str],
) -> None:
    assert len(dev) == len(holdout) == SPLIT_SIZE
    dev_dialogues = {item["source"]["dialogue_id"] for item in dev}
    holdout_dialogues = {item["source"]["dialogue_id"] for item in holdout}
    assert len(dev_dialogues) == len(dev)
    assert len(holdout_dialogues) == len(holdout)
    assert dev_dialogues <= exposed_ids
    assert not holdout_dialogues & exposed_ids
    assert not dev_dialogues & holdout_dialogues
    for items in (dev, holdout):
        actual = Counter(item["source"]["evaluation_category"] for item in items)
        assert actual == Counter(CATEGORY_TARGET)
        assert len({item["question_id"] for item in items}) == len(items)
        assert all(item["review"]["status"] == "pending" for item in items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--review-overrides", type=Path, default=DEFAULT_REVIEW_OVERRIDES)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args(argv)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    source_eval_path = args.source_root / "test/eval_turns.json"
    recorded = next(
        source for source in inventory["source_files"] if source["path"] == "test/eval_turns.json"
    )
    actual_hash = _sha256(source_eval_path)
    if actual_hash != recorded["sha256"]:
        raise ValueError(
            f"source eval hash mismatch: inventory={recorded['sha256']} actual={actual_hash}"
        )

    source_items = json.loads(source_eval_path.read_text(encoding="utf-8"))["items"]
    exposed_ids = set(inventory["holdout_excluded_dialogue_ids"])
    dev_pool = [item for item in source_items if item["dialogue_id"] in exposed_ids]
    holdout_pool = [item for item in source_items if item["dialogue_id"] not in exposed_ids]

    dev_source = _pick_split(
        dev_pool, split="dev", seed=args.seed, category_target=CATEGORY_TARGET
    )
    holdout_source = _pick_split(
        holdout_pool, split="holdout", seed=args.seed, category_target=CATEGORY_TARGET
    )
    dev = [_convert_candidate(item, "dev", i) for i, item in enumerate(dev_source, 1)]
    holdout = [
        _convert_candidate(item, "holdout", i) for i, item in enumerate(holdout_source, 1)
    ]
    _validate(dev, holdout, exposed_ids)
    review_overrides_sha256 = _apply_review_overrides(
        {"dev": dev, "holdout": holdout}, args.review_overrides
    )

    # The same fixed 20 development items serve as smoke and judge-calibration
    # candidates, avoiding an extra paid set. They still require label review.
    dev_by_id = {item["question_id"]: item for item in dev}
    smoke_source_pool = [
        item for item in dev_source if item["eval_item_id"] in dev_by_id
    ]
    smoke_source = _pick_split(
        smoke_source_pool,
        split="smoke",
        seed=args.seed,
        category_target=SMOKE_CATEGORY_TARGET,
    )
    smoke_ids = [item["eval_item_id"] for item in smoke_source]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_meta = {
        "candidate_release": "evaluation-v1-candidate-1",
        "status": "pending_human_label_review_not_for_scoring",
        "sampler_version": SAMPLER_VERSION,
        "seed": args.seed,
        "source_eval_sha256": actual_hash,
        "source_corpus_fingerprint_sha256": inventory["corpus_fingerprint_sha256"],
        "manual_review_overrides_sha256": review_overrides_sha256,
        "category_target": CATEGORY_TARGET,
        "selection_policy": {
            "one_target_turn_per_dialogue": True,
            "dev_dialogue_pool": "known historically exposed dialogue IDs",
            "holdout_dialogue_pool": "all source dialogues excluding known exposed IDs",
            "balancing_dimensions": [
                "domain",
                "dialogue_language",
                "language_relation",
                "history_bucket",
                "document_format",
            ],
        },
    }
    _write_json(args.out_dir / "dev_candidates.json", {**source_meta, "items": dev})
    _write_json(
        args.out_dir / "holdout_candidates.json", {**source_meta, "items": holdout}
    )
    _write_review_csv(args.out_dir / "dev_review.csv", dev)
    _write_review_csv(args.out_dir / "holdout_review.csv", holdout)

    split_manifest = {
        **source_meta,
        "counts": {
            "source_items": len(source_items),
            "exposed_dialogues": len(exposed_ids),
            "dev_pool_items": len(dev_pool),
            "holdout_pool_items": len(holdout_pool),
            "dev_items": len(dev),
            "holdout_items": len(holdout),
            "smoke_and_judge_calibration_items": len(smoke_ids),
        },
        "distribution": {"dev": _distribution(dev), "holdout": _distribution(holdout)},
        "dev_question_ids": [item["question_id"] for item in dev],
        "holdout_question_ids": [item["question_id"] for item in holdout],
        "smoke_and_judge_calibration_question_ids": smoke_ids,
        "validation": {
            "dev_holdout_dialogue_overlap": 0,
            "holdout_exposed_dialogue_overlap": 0,
            "duplicate_dialogues_within_each_split": 0,
            "source_hash_verified": True,
        },
    }
    _write_json(args.out_dir / "split_manifest.json", split_manifest)

    print(json.dumps(split_manifest["counts"], ensure_ascii=False))
    print(json.dumps(split_manifest["distribution"], ensure_ascii=False))
    print(f"Wrote pending-review candidates to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
