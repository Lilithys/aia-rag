"""Metrics for comparing chunking strategies/configs.

Three families of metric, from cheapest/most-direct to most end-to-end:

  1. Structural stats -- chunk count and token-size distribution. A direct
     proxy for indexing/embedding cost (num_chunks ~ vector count) and for
     how consistently sized the chunks are.

  2. Evidence integrity -- uses the test set's own gold evidence spans
     directly, no retriever involved: does the exact answer-supporting text
     for each question end up sitting whole inside a single chunk? This is
     the most direct measure of whether a chunking strategy is *capable* of
     supporting a correct, grounded answer, independent of retrieval
     quality.

  3. BM25 retrieval recall/MRR -- a lightweight, embedding-model-agnostic
     retrieval proxy (no GPU/download needed) to see how chunk boundaries
     affect actual retrievability under a real (if simple) retriever.
     Chunking's effect on retrievability is largely retriever-agnostic: a
     chunk that splits the answer across a boundary hurts BM25 and a dense
     retriever similarly. The final embedding-based retriever is a separate,
     later decision and will be evaluated on its own.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .eval_utils import Bm25Index, evidence_found_in_chunk
from .strategies.base import ChunkRecord


def structural_stats(chunks: list[ChunkRecord]) -> dict:
    sizes = sorted(c.token_count for c in chunks)
    n = len(sizes)
    if n == 0:
        return {"num_chunks": 0, "total_tokens": 0}

    def pct(p):
        return sizes[min(n - 1, int(p * n))]

    return {
        "num_chunks": n,
        "total_tokens": sum(sizes),
        "avg_tokens": round(sum(sizes) / n, 1),
        "median_tokens": pct(0.5),
        "p10_tokens": pct(0.10),
        "p90_tokens": pct(0.90),
        "min_tokens": sizes[0],
        "max_tokens": sizes[-1],
    }


def required_doc_evidence(item: dict) -> dict[str, list[str]]:
    """doc_id -> list of distinct gold evidence_text strings for that doc."""
    out = {}
    for ev in item.get("evidence", []):
        texts = sorted({s["evidence_text"] for s in ev.get("spans", []) if s.get("evidence_text")})
        if texts:
            out[ev["doc_id"]] = texts
    return out


def evidence_integrity(chunks_by_doc: dict[str, list[ChunkRecord]], test_items: list[dict]) -> dict:
    span_total = 0
    span_found = 0
    doc_total = 0
    doc_single_chunk_covered = 0

    for item in test_items:
        if not item.get("answerable", True):
            continue
        for doc_id, texts in required_doc_evidence(item).items():
            chunks = chunks_by_doc.get(doc_id, [])
            doc_total += 1
            hits_per_chunk = Counter()
            for text in texts:
                span_total += 1
                found = False
                for c in chunks:
                    if evidence_found_in_chunk(text, c.text):
                        found = True
                        hits_per_chunk[c.chunk_id] += 1
                if found:
                    span_found += 1
            if hits_per_chunk and max(hits_per_chunk.values()) == len(texts):
                doc_single_chunk_covered += 1

    return {
        "evidence_span_containment_rate": round(span_found / span_total, 4) if span_total else None,
        "evidence_span_total": span_total,
        "single_chunk_full_coverage_rate": round(doc_single_chunk_covered / doc_total, 4) if doc_total else None,
        "required_doc_total": doc_total,
    }


def query_text(item: dict) -> str:
    turns = [t["utterance"] for t in item.get("conversation_history", [])]
    return "\n".join(turns + [item["question"]])


def retrieval_eval_per_item(index, test_items: list[dict], k_values=(5, 10), query_fn=query_text) -> list[dict]:
    """Runs retrieval for every answerable question against `index` (any
    object exposing `.chunks` and `.ranked_indices(query) -> list[int]` --
    both Bm25Index and embedding.index.DenseIndex satisfy this) and returns
    one raw result record per question, so callers can aggregate overall
    and/or sliced by a field like `language_relation` without re-running
    retrieval.

    `query_fn(item) -> str` builds the retrieval query; defaults to the
    naive conversation-history-concatenation used throughout steps 2-4.
    Passing a different query_fn (e.g. a query-rewriting step) is how the
    query-rewriting A/B test reuses this exact harness."""
    max_k = max(k_values)
    results = []

    for item in test_items:
        if not item.get("answerable", True):
            continue
        required = required_doc_evidence(item)
        if not required:
            continue
        ranked = index.ranked_indices(query_fn(item))
        top_chunks = [index.chunks[i] for i in ranked[:max_k]]

        record = {
            "question_id": item.get("question_id"),
            "language_relation": item.get("language_relation"),
            "question_type": item.get("question_type"),
            "recall_hit": {},
            "multidoc_full_hit": {},
            "precision": {},
        }
        for k in k_values:
            window = top_chunks[:k]
            covered_docs = 0
            for doc_id, texts in required.items():
                if any(evidence_found_in_chunk(t, c.text) for t in texts for c in window if c.doc_id == doc_id):
                    covered_docs += 1
            record["recall_hit"][k] = covered_docs > 0
            record["multidoc_full_hit"][k] = covered_docs == len(required)
            if window:
                relevant = sum(
                    1
                    for c in window
                    if c.doc_id in required and any(evidence_found_in_chunk(t, c.text) for t in required[c.doc_id])
                )
                record["precision"][k] = relevant / len(window)

        rr = 0.0
        for rank, c in enumerate(top_chunks, start=1):
            texts = required.get(c.doc_id)
            if texts and any(evidence_found_in_chunk(t, c.text) for t in texts):
                rr = 1.0 / rank
                break
        record["rr"] = rr
        results.append(record)
    return results


def aggregate_retrieval_results(per_item: list[dict], k_values=(5, 10)) -> dict:
    max_k = max(k_values)
    n = len(per_item)
    out = {
        "mrr_at_%d" % max_k: round(sum(r["rr"] for r in per_item) / n, 4) if n else None,
        "n_answerable_questions": n,
    }
    for k in k_values:
        recalls = [r["recall_hit"][k] for r in per_item]
        multidoc = [r["multidoc_full_hit"][k] for r in per_item]
        precisions = [r["precision"][k] for r in per_item if k in r["precision"]]
        out[f"recall_at_{k}"] = round(sum(recalls) / len(recalls), 4) if recalls else None
        out[f"multidoc_full_recall_at_{k}"] = round(sum(multidoc) / len(multidoc), 4) if multidoc else None
        out[f"context_precision_at_{k}"] = round(sum(precisions) / len(precisions), 4) if precisions else None
    return out


def retrieval_eval(all_chunks: list[ChunkRecord], test_items: list[dict], k_values=(5, 10)) -> dict:
    index = Bm25Index(all_chunks)
    per_item = retrieval_eval_per_item(index, test_items, k_values)
    return aggregate_retrieval_results(per_item, k_values)


def group_by_doc(chunks: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    out = defaultdict(list)
    for c in chunks:
        out[c.doc_id].append(c)
    return dict(out)
