"""Dense cosine-similarity index -- same shape as chunking's Bm25Index
(`.chunks`, `.ranked_indices(query)`) so it drops into the existing
`src/chunking/evaluate.py::retrieval_eval` unchanged.
"""
from __future__ import annotations

import numpy as np

from src.chunking.strategies.base import ChunkRecord

from .models.base import EmbeddingModel


class DenseIndex:
    def __init__(
        self,
        chunks: list[ChunkRecord],
        model: EmbeddingModel,
        passage_embeddings: np.ndarray,
        query_embedding_cache: dict[str, np.ndarray] | None = None,
    ):
        self.chunks = chunks
        self.model = model
        # embeddings are pre-normalized (each backend L2-normalizes), so dot
        # product == cosine similarity
        self._embeddings = passage_embeddings
        # test-set queries are known upfront, so run_experiment batches them
        # through encode_queries() once instead of one call per question --
        # this cache is just that lookup, with a per-call fallback for
        # anything not pre-embedded.
        self._cache = query_embedding_cache or {}

    def ranked_indices(self, query: str) -> list[int]:
        q = self._cache.get(query)
        if q is None:
            q = self.model.encode_queries([query])[0]
        scores = self._embeddings @ q
        return list(np.argsort(-scores))
