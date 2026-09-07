"""Compare the frozen and heading-preserving indexes on dev retrieval only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from src.artifacts import file_sha256, fingerprint
from src.embedding.embed_corpus import load_chunks
from src.evaluation.evidence import answer_targets, covered_targets, score_window
from src.evaluation.paid_calls_v1 import atomic_json
from src.retrieval.folder_retriever import FolderRetriever


PROFILES = ("baseline_dense_k10", "rerank_windowed_k5")


def summarize(rows: list[dict]) -> dict:
    return {
        "answer_n": len(rows),
        "any_full_block_hit": sum(r["any_answer_evidence_hit"] for r in rows),
        "all_full_blocks_hit": sum(r["all_annotated_answer_blocks_hit"] for r in rows),
        "mean_block_recall": statistics.mean(r["annotated_answer_block_recall"] for r in rows),
        "mean_evidence_precision": statistics.mean(r["annotated_evidence_precision"] for r in rows),
        "mean_context_tokens": statistics.mean(r["context_tokens"] for r in rows),
        "p50_retrieval_seconds": statistics.median(r["retrieval_seconds"] for r in rows),
    }


def evaluate(folder: Path, items: list[dict]) -> tuple[dict, dict]:
    retriever = FolderRetriever(folder)
    all_chunks = load_chunks(str(folder / "chunks.jsonl"))
    index_targets = covered_targets(all_chunks, {
        target for item in items for target in answer_targets(item)
    })
    by_profile: dict[str, list[dict]] = {name: [] for name in PROFILES}
    for item in items:
        targets = answer_targets(item)
        for profile in PROFILES:
            chunks, timing = retriever.retrieve(item, profile)
            row = {
                "question_id": item["question_id"],
                "document_format": item["document_format"],
                "question_language": item["question_language"],
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "retrieval_seconds": timing["retrieval_total_seconds"],
                **score_window(chunks, targets),
            }
            by_profile[profile].append(row)
    unique_targets = {target for item in items for target in answer_targets(item)}
    return {
        "folder": str(folder),
        "chunks": len(all_chunks),
        "setup_seconds": retriever.setup_seconds,
        "unique_answer_blocks": len(unique_targets),
        "answer_blocks_represented_in_index": len(index_targets),
        "profiles": {name: summarize(rows) for name, rows in by_profile.items()},
    }, by_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=Path("doc/evaluation/v1/releases/v1.1/dev.json"))
    parser.add_argument("--old", type=Path, default=Path("data/experiments/eval_v1/retrieval_fixed/semantic_1024"))
    parser.add_argument("--new", type=Path, default=Path("data/experiments/eval_v12/delivery_candidate/semantic_1024_v2"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/eval_v12/delivery_candidate/retrieval_comparison.json"))
    args = parser.parse_args()
    items = [item for item in json.loads(args.dev.read_text())["items"] if item["task_label"] == "answer"]
    plan = {
        "study": "heading-preservation-dev-retrieval-v1",
        "split": "dev-answer-only",
        "question_ids": [item["question_id"] for item in items],
        "profiles": list(PROFILES),
        "selection_rule": "Adopt only if the repaired target becomes retrievable without reducing aggregate full-block hits.",
        "holdout_queried": False,
        "paid_api_calls": 0,
        "inputs": {str(path): file_sha256(path) for path in (
            args.dev,
            args.old / "chunks.jsonl",
            args.old / "embeddings/bge_m3/embedding_manifest.json",
            args.new / "chunks.jsonl",
            args.new / "embeddings/bge_m3/embedding_manifest.json",
        )},
        "code_sha256": {str(path): file_sha256(path) for path in (
            Path(__file__), Path("src/retrieval/folder_retriever.py"),
            Path("src/retrieval/candidates_v1.py"), Path("src/evaluation/evidence.py"),
        )},
    }
    summaries, rows = {}, {}
    for label, folder in (("frozen_v1", args.old), ("heading_preserving_v2", args.new)):
        summaries[label], rows[label] = evaluate(folder, items)
        print(label, json.dumps(summaries[label]["profiles"], ensure_ascii=False), flush=True)
    old, new = rows["frozen_v1"], rows["heading_preserving_v2"]
    paired = {}
    for profile in PROFILES:
        old_by_id = {row["question_id"]: row for row in old[profile]}
        changes = []
        for row in new[profile]:
            before = old_by_id[row["question_id"]]
            if before["all_annotated_answer_blocks_hit"] != row["all_annotated_answer_blocks_hit"]:
                changes.append({
                    "question_id": row["question_id"],
                    "old": before["all_annotated_answer_blocks_hit"],
                    "new": row["all_annotated_answer_blocks_hit"],
                })
        paired[profile] = {
            "gained": sum(change["new"] > change["old"] for change in changes),
            "lost": sum(change["new"] < change["old"] for change in changes),
            "changes": changes,
        }
    result = {
        "plan": plan,
        "plan_binding_sha256": fingerprint(plan),
        "summaries": summaries,
        "paired": paired,
        "items": rows,
        "recommendation": "pending",
    }
    for profile in PROFILES:
        result["summaries"]["heading_preserving_v2"]["profiles"][profile]["status_counts"] = dict(Counter(
            "hit" if row["all_annotated_answer_blocks_hit"] else "miss" for row in rows["heading_preserving_v2"][profile]
        ))
    atomic_json(args.out, result)
    print(f"Wrote {args.out}; no paid API calls; holdout not loaded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
