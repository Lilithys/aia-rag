"""Dense (Chroma), sparse (BM25), and RRF hybrid retrievers -- all exposing
the same `.chunks` + `.ranked_indices(query) -> list[int]` interface used by
`src/chunking/evaluate.py::retrieval_eval_per_item` since step 2, so the
exact same evaluation harness applies unchanged here too.
"""
from __future__ import annotations

import threading
from collections import defaultdict

from src.chunking.strategies.base import ChunkRecord
from src.embedding.models.base import EmbeddingModel


class ChromaDenseIndex:
    def __init__(self, chunks: list[ChunkRecord], collection, model: EmbeddingModel):
        self.chunks = chunks
        self._collection = collection
        self._model = model
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        # sentence-transformers/torch inference isn't guaranteed safe to call
        # concurrently from multiple threads on the same model instance; this
        # only serializes the (fast) embedding step, not the (slow) LLM calls
        # callers typically run alongside it, so it costs little under
        # concurrent evaluation.
        self._lock = threading.Lock()

    def ranked_indices(self, query: str) -> list[int]:
        return [i for i, _score in self.ranked_with_scores(query, len(self.chunks))]

    def ranked_with_scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Chroma's collection is configured for cosine space, so
        distance = 1 - cosine_similarity; we convert back to similarity
        here since that's the number the confidence threshold reasons about."""
        with self._lock:
            q_emb = self._model.encode_queries([query])[0]
            result = self._collection.query(
                query_embeddings=[q_emb.tolist()], n_results=top_k, include=["distances"]
            )
        return [
            (self._id_to_pos[cid], 1 - dist)
            for cid, dist in zip(result["ids"][0], result["distances"][0])
        ]


class RrfHybridIndex:
    """Reciprocal Rank Fusion: score(doc) = sum over each ranker of
    1 / (k_rrf + rank). Standard, parameter-light (k_rrf=60 is the value
    from the original RRF paper), needs no score normalization across
    dense cosine similarity and BM25's unbounded scores."""

    def __init__(self, chunks: list[ChunkRecord], dense_index, sparse_index, k_rrf: int = 60):
        self.chunks = chunks
        self._dense = dense_index
        self._sparse = sparse_index
        self._k_rrf = k_rrf

    def ranked_indices(self, query: str) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(self._dense.ranked_indices(query), start=1):
            scores[idx] += 1.0 / (self._k_rrf + rank)
        for rank, idx in enumerate(self._sparse.ranked_indices(query), start=1):
            scores[idx] += 1.0 / (self._k_rrf + rank)
        return sorted(scores, key=lambda i: -scores[i])
