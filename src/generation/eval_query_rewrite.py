"""Query-rewriting A/B test: does condensing multi-turn history into a
standalone query beat the naive concatenation used everywhere else in this
project? Reuses the exact retrieval harness from steps 2-4
(`src/chunking/evaluate.py`) via its `query_fn` parameter -- same metrics,
same retriever (dense/Chroma+BGE-M3, the step-4 winner), only the query
construction differs between the two runs.

Usage:
    uv run python -m src.generation.eval_query_rewrite
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.chunking.evaluate import aggregate_retrieval_results, query_text, retrieval_eval_per_item
from src.retrieval.retriever import get_default_retriever

from .providers import REWRITE_MODEL
from .query_rewrite import QueryRewriter

OUT_PATH = "data/processed/query_rewrite_experiment_results.json"
K_VALUES = (5, 10)
# a single local Ollama server, not a hosted API with elastic capacity
MAX_WORKERS = 3


def load_test_items(path: str = "doc/test/test.json") -> list[dict]:
    return json.load(open(path, encoding="utf-8"))["items"]


def precompute_rewrites(items: list[dict], rewriter: QueryRewriter) -> dict[str, str]:
    answerable = [it for it in items if it.get("answerable", True)]
    out: dict[str, str] = {}

    def _one(item):
        return item["question_id"], rewriter.rewrite(item["conversation_history"], item["question"])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_one, it): it for it in answerable}
        for i, fut in enumerate(as_completed(futures), 1):
            qid, rewritten = fut.result()
            out[qid] = rewritten
            print(f"  [{i}/{len(answerable)}] rewritten", file=sys.stderr)
    return out


def main(argv=None) -> int:
    items = load_test_items()
    answerable = [it for it in items if it.get("answerable", True)]
    print(f"Loaded {len(items)} test questions ({len(answerable)} answerable).", file=sys.stderr)

    print(f"Rewriting queries ({REWRITE_MODEL})...", file=sys.stderr)
    rewriter = QueryRewriter()
    t0 = time.time()
    rewrites = precompute_rewrites(items, rewriter)
    rewrite_wall_s = time.time() - t0
    print(f"  done in {rewrite_wall_s:.1f}s", file=sys.stderr)

    retriever = get_default_retriever()
    index = retriever.index  # retrieval_eval_per_item expects the lower-level index

    def rewritten_query_fn(item: dict) -> str:
        return rewrites.get(item["question_id"], query_text(item))

    print("Baseline (naive concat) retrieval eval...", file=sys.stderr)
    baseline_per_item = retrieval_eval_per_item(index, items, K_VALUES, query_fn=query_text)
    baseline = aggregate_retrieval_results(baseline_per_item, K_VALUES)

    print("Rewritten-query retrieval eval...", file=sys.stderr)
    rewritten_per_item = retrieval_eval_per_item(index, items, K_VALUES, query_fn=rewritten_query_fn)
    rewritten_agg = aggregate_retrieval_results(rewritten_per_item, K_VALUES)

    # breakdown: does rewriting matter more for questions that actually have
    # conversation history (where there's something to condense) vs single-turn?
    def _breakdown(per_item, items_by_id):
        with_hist, without_hist = [], []
        for r in per_item:
            it = items_by_id[r["question_id"]]
            (with_hist if it.get("conversation_history") else without_hist).append(r)
        return {
            "with_history": aggregate_retrieval_results(with_hist, K_VALUES) if with_hist else None,
            "without_history": aggregate_retrieval_results(without_hist, K_VALUES) if without_hist else None,
            "n_with_history": len(with_hist),
            "n_without_history": len(without_hist),
        }

    items_by_id = {it["question_id"]: it for it in items}
    baseline_breakdown = _breakdown(baseline_per_item, items_by_id)
    rewritten_breakdown = _breakdown(rewritten_per_item, items_by_id)

    out = {
        "baseline": baseline,
        "rewritten": rewritten_agg,
        "baseline_breakdown": baseline_breakdown,
        "rewritten_breakdown": rewritten_breakdown,
        "rewrite_wall_seconds": round(rewrite_wall_s, 1),
        "rewrite_wall_seconds_per_query": round(rewrite_wall_s / len(answerable), 2),
        "sample_rewrites": [
            {"question_id": it["question_id"], "original_question": it["question"], "rewritten": rewrites[it["question_id"]]}
            for it in answerable[:5]
        ],
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nbaseline:  {baseline}", file=sys.stderr)
    print(f"rewritten: {rewritten_agg}", file=sys.stderr)
    print(f"\nWrote results to {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
