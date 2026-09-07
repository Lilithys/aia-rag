"""Executable frozen retrieval candidates for the v1 generation comparison."""
from __future__ import annotations

import time
from pathlib import Path

from src.evaluation.evidence import retrieval_query

CANDIDATES = {
    "baseline_dense_k10": {"top_k": 10, "rerank": None},
    "rerank_truncated_k10": {"top_k": 10, "rerank": "truncated"},
    "rerank_windowed_k5": {"top_k": 5, "rerank": "windowed"},
}
RERANK_CONFIG = {"model": "BAAI/bge-reranker-base", "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
                 "candidate_k": 20, "query_tail_tokens": 96, "passage_window_tokens": 384,
                 "stride": 320, "max_pair_tokens": 512, "batch_size": 16, "parent_score": "max"}


class CandidateRetriever:
    def __init__(self, artifact_root: Path):
        import torch
        from sentence_transformers import CrossEncoder
        from src.retrieval.retriever import get_default_retriever
        torch.set_num_threads(4)
        started = time.perf_counter()
        folder = artifact_root / "semantic_1024"
        self.dense = get_default_retriever(str(folder / "chunks.jsonl"),
            str(folder / "embeddings/bge_m3/chunk_embeddings.npy"), str(folder / "chroma"))
        self.reranker = CrossEncoder(RERANK_CONFIG["model"], revision=RERANK_CONFIG["revision"], local_files_only=True,
                                     processor_kwargs={"model_max_length": 512})
        self.reranker.predict([["query", "passage"]], show_progress_bar=False)
        self.dense.search("user: What services are available?", top_k=20)
        self.setup_seconds = time.perf_counter() - started

    def retrieve(self, item, name):
        config = CANDIDATES[name]
        started = time.perf_counter()
        query = retrieval_query(item)
        scored = self.dense.search_with_scores(query, top_k=RERANK_CONFIG["candidate_k"])
        chunks = [chunk for chunk, _ in scored]
        metadata = {"dense_seconds": time.perf_counter() - started, "rerank_seconds": 0.0,
                    "top1_dense_score": float(scored[0][1]) if scored else 0.0,
                    "candidate_ids": [c.chunk_id for c in chunks], "query": query,
                    "pairs_exceeding_limit_before_truncation": 0}
        if config["rerank"] and chunks:
            rerank_start = time.perf_counter()
            tok = self.reranker.tokenizer
            query_ids = tok.encode(query, add_special_tokens=False)
            query = tok.decode(query_ids[-RERANK_CONFIG["query_tail_tokens"]:])
            pairs, owners = [], []
            for i, chunk in enumerate(chunks):
                if config["rerank"] == "truncated":
                    passages = [chunk.text]
                else:
                    ids = tok.encode(chunk.text, add_special_tokens=False)
                    passages = [tok.decode(ids[j:j + RERANK_CONFIG["passage_window_tokens"]])
                                for j in range(0, len(ids), RERANK_CONFIG["stride"])] or [""]
                for passage in passages:
                    oversized = len(tok.encode(query, passage, add_special_tokens=True, truncation=False)) > RERANK_CONFIG["max_pair_tokens"]
                    metadata["pairs_exceeding_limit_before_truncation"] += int(oversized)
                    if oversized and config["rerank"] == "windowed":
                        raise ValueError("Windowed reranker input exceeded its fixed limit")
                    pairs.append([query, passage]); owners.append(i)
            scores = self.reranker.predict(pairs, batch_size=RERANK_CONFIG["batch_size"], show_progress_bar=False, convert_to_numpy=True)
            best = {i: -float("inf") for i in range(len(chunks))}
            for owner, score in zip(owners, scores, strict=True):
                best[owner] = max(best[owner], float(score))
            chunks = [chunks[i] for i in sorted(best, key=lambda i: (-best[i], i))]
            metadata.update(rerank_seconds=time.perf_counter() - rerank_start, pairs_scored=len(pairs),
                            query_shortened=len(query_ids) > RERANK_CONFIG["query_tail_tokens"])
        metadata["retrieval_total_seconds"] = time.perf_counter() - started
        return chunks[:config["top_k"]], metadata
