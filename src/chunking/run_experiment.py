"""Step 2b: sweep {strategy x chunk_size x overlap} and compare.

Usage:
    uv run python -m src.chunking.run_experiment

Writes data/processed/chunking_experiment_results.json and .csv, one row
per (strategy, chunk_size, overlap) combination, with structural stats,
evidence-integrity metrics, and BM25 retrieval metrics for each.
"""
from __future__ import annotations

import csv
import json
import sys
import time

from .evaluate import evidence_integrity, group_by_doc, retrieval_eval, structural_stats
from .loader import load_clean_corpus
from .strategies import fixed_size, recursive_structural, semantic_section

STRATEGIES = {
    "A_fixed_size": fixed_size,
    "B_recursive_structural": recursive_structural,
    "C_semantic_section": semantic_section,
}

CHUNK_SIZES = [256, 512, 1024]
OVERLAP_RATIOS = [0.0, 0.15, 0.3]


def run_one(mod, docs, test_items, chunk_size: int, overlap: int) -> dict:
    t0 = time.time()
    chunks = []
    for d in docs:
        chunks.extend(mod.chunk_document(d.body, d.meta, chunk_size=chunk_size, overlap=overlap))
    stats = structural_stats(chunks)
    ei = evidence_integrity(group_by_doc(chunks), test_items)
    ret = retrieval_eval(chunks, test_items)
    elapsed = round(time.time() - t0, 2)
    return {**stats, **ei, **ret, "seconds": elapsed}


def main(argv=None) -> int:
    docs = load_clean_corpus()
    test = json.load(open("doc/test/test.json", encoding="utf-8"))
    items = test["items"]
    print(f"Loaded {len(docs)} documents, {len(items)} test questions.", file=sys.stderr)

    rows = []
    total = len(STRATEGIES) * len(CHUNK_SIZES) * len(OVERLAP_RATIOS)
    i = 0
    for strategy_name, mod in STRATEGIES.items():
        for chunk_size in CHUNK_SIZES:
            for ratio in OVERLAP_RATIOS:
                overlap = round(chunk_size * ratio)
                i += 1
                print(
                    f"[{i}/{total}] {strategy_name} chunk_size={chunk_size} overlap={overlap} ({ratio:.0%})",
                    file=sys.stderr,
                )
                result = run_one(mod, docs, items, chunk_size, overlap)
                rows.append(
                    {
                        "strategy": strategy_name,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "overlap_ratio": ratio,
                        **result,
                    }
                )

    out_json = "data/processed/chunking_experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    out_csv = "data/processed/chunking_experiment_results.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_json} and {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
