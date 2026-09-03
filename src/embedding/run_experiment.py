"""Step 3: compare embedding models on the materialized chunk set.

Reuses the exact same evidence-matching + retrieval-eval harness the
chunking comparison used (src/chunking/evaluate.py) -- only the retriever
changes (BM25 -> dense cosine over each model's embeddings), which is the
whole point: it isolates the embedding model as the only variable.

Usage:
    uv run python -m src.embedding.run_experiment
    uv run python -m src.embedding.run_experiment --models bge_m3 multilingual_e5 openai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from src.chunking.eval_utils import Bm25Index
from src.chunking.evaluate import aggregate_retrieval_results, query_text, retrieval_eval_per_item
from src.chunking.tokenizer import count_tokens

from .embed_corpus import embed_chunks_cached, load_chunks
from .index import DenseIndex
from .models import REGISTRY

K_VALUES = (5, 10)


def load_test_items(path: str = "doc/test/test.json") -> list[dict]:
    return json.load(open(path, encoding="utf-8"))["items"]


def breakdown_by(per_item: list[dict], field: str, k_values=K_VALUES) -> dict:
    groups = defaultdict(list)
    for r in per_item:
        groups[r.get(field)].append(r)
    return {str(key): {"n": len(rs), **aggregate_retrieval_results(rs, k_values)} for key, rs in groups.items()}


def run_bm25_baseline(chunks, items, k_values=K_VALUES):
    index = Bm25Index(chunks)
    per_item = retrieval_eval_per_item(index, items, k_values)
    return per_item, {
        "model": "bm25_baseline",
        "dim": None,
        "is_local": True,
        **aggregate_retrieval_results(per_item, k_values),
    }


def run_embedding_model(model_key: str, chunks, items, k_values=K_VALUES):
    model = REGISTRY[model_key]()

    corpus_tokens = sum(count_tokens(c.text) for c in chunks)
    t0 = time.time()
    passage_emb = embed_chunks_cached(model, chunks)
    embed_corpus_s = time.time() - t0

    answerable_items = [it for it in items if it.get("answerable", True)]
    query_texts = [query_text(it) for it in answerable_items]
    query_tokens = sum(count_tokens(q) for q in query_texts)
    t0 = time.time()
    query_embs = model.encode_queries(query_texts)
    embed_query_s = time.time() - t0
    query_cache = dict(zip(query_texts, query_embs))

    index = DenseIndex(chunks, model, passage_emb, query_cache)
    per_item = retrieval_eval_per_item(index, items, k_values)

    result = {
        "model": model_key,
        "dim": model.dim,
        "is_local": model.is_local,
        "embed_corpus_seconds": round(embed_corpus_s, 2),
        "embed_query_seconds": round(embed_query_s, 2),
        "corpus_tokens_per_sec": round(corpus_tokens / embed_corpus_s, 1) if embed_corpus_s > 0 else None,
        "query_tokens_per_sec": round(query_tokens / embed_query_s, 1) if embed_query_s > 0 else None,
        **aggregate_retrieval_results(per_item, k_values),
    }
    if hasattr(model, "estimated_cost_usd"):
        result["estimated_cost_usd"] = round(model.estimated_cost_usd(), 4)
        result["estimated_cost_per_1k_calls_usd"] = round(model.estimated_cost_usd() / len(query_texts) * 1000, 2)
    return per_item, result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["bge_m3", "multilingual_e5"], choices=list(REGISTRY))
    parser.add_argument("--out", default="data/processed/embedding_experiment_results.json")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    items = load_test_items()
    print(f"Loaded {len(chunks)} chunks, {len(items)} test questions.", file=sys.stderr)

    results = []
    breakdowns = {}

    print("[baseline] BM25 (from step 2, for reference)", file=sys.stderr)
    bm25_per_item, bm25_result = run_bm25_baseline(chunks, items)
    results.append(bm25_result)
    breakdowns["bm25_baseline"] = breakdown_by(bm25_per_item, "language_relation")

    for model_key in args.models:
        print(f"[model] {model_key}", file=sys.stderr)
        t0 = time.time()
        per_item, result = run_embedding_model(model_key, chunks, items)
        print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)
        results.append(result)
        breakdowns[model_key] = breakdown_by(per_item, "language_relation")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "breakdown_by_language_relation": breakdowns}, f, ensure_ascii=False, indent=2)

    print(f"\nWrote results to {args.out}", file=sys.stderr)
    for r in results:
        print(
            f"  {r['model']:20s} dim={str(r['dim']):5s} recall@5={r['recall_at_5']} "
            f"recall@10={r['recall_at_10']} mrr@10={r['mrr_at_10']} prec@5={r['context_precision_at_5']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
