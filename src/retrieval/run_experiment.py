"""Step 4: build the actual retriever and compare strategies.

Stage 1 -- Dense (Chroma + BGE-M3) vs Sparse (BM25) vs Hybrid (RRF fusion),
scored on Recall@5/10, MRR@10, Context Precision@5/10 (same harness as
steps 2-3: src/chunking/evaluate.py).

Stage 2 -- reranker on/off on whichever of the three wins stage 1, using
BAAI/bge-reranker-base over each winner's top-30 candidates. Reports
Recall@5, Context Precision@5, and latency (a local cross-encoder has no $
cost, so latency stands in for both, per the discussion behind this step).

Stage 3 -- a top_k sensitivity sweep (k=1,3,5,10,20) on the final winning
config (best strategy + reranker if it won stage 2), the analysis
require.md asks for directly.

Usage:
    uv run python -m src.retrieval.run_experiment
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from src.chunking.eval_utils import Bm25Index
from src.chunking.evaluate import aggregate_retrieval_results, retrieval_eval_per_item
from src.embedding.embed_corpus import load_chunks
from src.embedding.models.bge_m3 import BgeM3Model

from .chroma_store import get_collection
from .indexes import ChromaDenseIndex, RrfHybridIndex
from .reranker import BgeReranker, RerankIndex

K_STAGE1 = (5, 10)
K_SWEEP = (1, 3, 5, 10, 20)
OUT_PATH = "data/processed/retrieval_experiment_results.json"


def load_test_items(path: str = "doc/test/test.json") -> list[dict]:
    return json.load(open(path, encoding="utf-8"))["items"]


def timed_eval(index, items: list[dict], k_values: tuple[int, ...]) -> tuple[dict, float]:
    """Runs retrieval_eval_per_item once, timed, returning (aggregated
    metrics, avg wall-clock ms per question) -- so a reranker's cost is
    measured on the exact same pass that produces its quality numbers,
    not a separate, redundant run."""
    t0 = time.time()
    per_item = retrieval_eval_per_item(index, items, k_values)
    elapsed_ms = (time.time() - t0) * 1000
    avg_ms = elapsed_ms / len(per_item) if per_item else 0.0
    return aggregate_retrieval_results(per_item, k_values), avg_ms


def main(argv=None) -> int:
    chunks = load_chunks()
    items = load_test_items()
    print(f"Loaded {len(chunks)} chunks, {len(items)} test questions.", file=sys.stderr)

    print("Loading BGE-M3 + cached corpus embeddings...", file=sys.stderr)
    model = BgeM3Model()
    embeddings = np.load("data/processed/embeddings/bge_m3/chunk_embeddings.npy")
    collection = get_collection(chunks, embeddings)

    dense = ChromaDenseIndex(chunks, collection, model)
    sparse = Bm25Index(chunks)
    hybrid = RrfHybridIndex(chunks, dense, sparse)

    # ---------- stage 1: dense vs sparse vs hybrid ----------
    print("\n[stage 1] dense vs sparse vs hybrid", file=sys.stderr)
    stage1 = {}
    strategies = {"dense": dense, "sparse": sparse, "hybrid": hybrid}
    for name, index in strategies.items():
        per_item = retrieval_eval_per_item(index, items, K_STAGE1)
        stage1[name] = aggregate_retrieval_results(per_item, K_STAGE1)
        print(f"  {name:8s} {stage1[name]}", file=sys.stderr)

    winner_name = max(stage1, key=lambda n: stage1[n]["mrr_at_10"])
    winner_index = strategies[winner_name]
    print(f"  winner (by MRR@10): {winner_name}", file=sys.stderr)

    # ---------- stage 2 + 3: one timed pass per config, over the full k-sweep ----------
    # (K_SWEEP includes k=5, so stage 2's on/off table and stage 3's sweep come
    # from the same pass -- no need to rerank the same queries three times.)
    print(f"\n[stage 2+3] reranker on/off + top_k sweep on '{winner_name}'", file=sys.stderr)
    reranker = BgeReranker()
    reranked_index = RerankIndex(chunks, winner_index, reranker, candidate_n=30)

    sweep_no_rerank, latency_no_rerank = timed_eval(winner_index, items, K_SWEEP)
    sweep_rerank, latency_rerank = timed_eval(reranked_index, items, K_SWEEP)
    print(f"  no_reranker: latency={latency_no_rerank:.1f}ms {sweep_no_rerank}", file=sys.stderr)
    print(f"  reranker:    latency={latency_rerank:.1f}ms {sweep_rerank}", file=sys.stderr)

    # Whether the reranker's quality gain is worth its latency cost is exactly
    # the open question from the planning discussion (drop it if latency is
    # too high) -- a judgment call for the report, not baked in here.
    stage2 = {
        "base_strategy": winner_name,
        "no_reranker": {
            "recall_at_5": sweep_no_rerank["recall_at_5"],
            "context_precision_at_5": sweep_no_rerank["context_precision_at_5"],
            "latency_ms": round(latency_no_rerank, 1),
        },
        "reranker": {
            "recall_at_5": sweep_rerank["recall_at_5"],
            "context_precision_at_5": sweep_rerank["context_precision_at_5"],
            "latency_ms": round(latency_rerank, 1),
        },
    }

    out = {
        "stage1_strategy_comparison": stage1,
        "stage1_winner": winner_name,
        "stage2_reranker": stage2,
        "stage3_topk_sweep": {"no_reranker": sweep_no_rerank, "reranker": sweep_rerank},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote results to {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
