"""The retriever this project actually uses: dense-only (Chroma + BGE-M3),
no reranker -- per the step-4 comparison, hybrid RRF fusion and reranking
(both bge-reranker-base and -large) underperformed plain dense retrieval on
this corpus. See data/processed/retrieval_experiment_results.json.

Usage:
    from src.retrieval.retriever import get_default_retriever
    retriever = get_default_retriever()
    chunks = retriever.search("some question", top_k=10)
"""
from __future__ import annotations

import numpy as np

from src.chunking.strategies.base import ChunkRecord
from src.embedding.embed_corpus import load_chunks
from src.embedding.models.bge_m3 import BgeM3Model

from .chroma_store import get_collection
from .indexes import ChromaDenseIndex

DEFAULT_TOP_K = 10  # see step-4 report: recall@10 96.5% vs 85.9% at k=5, near the recall ceiling


class Retriever:
    def __init__(self, index: ChromaDenseIndex):
        self._index = index

    @property
    def index(self) -> ChromaDenseIndex:
        """The underlying .chunks + .ranked_indices(query) index, for
        callers that need to plug into the shared evaluation harness
        (src/chunking/evaluate.py::retrieval_eval_per_item) directly."""
        return self._index

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[ChunkRecord]:
        return [c for c, _score in self.search_with_scores(query, top_k)]

    def search_with_scores(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[ChunkRecord, float]]:
        pairs = self._index.ranked_with_scores(query, top_k)
        return [(self._index.chunks[i], score) for i, score in pairs]


def get_default_retriever(
    chunks_path: str = "data/processed/chunks/chunks.jsonl",
    embeddings_path: str = "data/processed/embeddings/bge_m3/chunk_embeddings.npy",
) -> Retriever:
    chunks = load_chunks(chunks_path)
    embeddings = np.load(embeddings_path)
    model = BgeM3Model()
    collection = get_collection(chunks, embeddings)
    return Retriever(ChromaDenseIndex(chunks, collection, model))
