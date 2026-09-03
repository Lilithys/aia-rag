"""Cross-encoder reranking on top of a base retriever's candidate list.

Standard retrieve-then-rerank: pull `candidate_n` from the base retriever
(cheap, wide net), score each (query, chunk) pair with a cross-encoder
(expensive but precise), re-sort. candidate_n must cover the largest k we'll
ever evaluate at.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.chunking.strategies.base import ChunkRecord

RERANKER_MODEL_ID = "BAAI/bge-reranker-base"


class BgeReranker:
    def __init__(self, model_id: str = RERANKER_MODEL_ID):
        self._model = CrossEncoder(model_id)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return list(self._model.predict([(query, t) for t in texts]))


class RerankIndex:
    def __init__(self, chunks: list[ChunkRecord], base_index, reranker: BgeReranker, candidate_n: int = 30):
        self.chunks = chunks
        self._base = base_index
        self._reranker = reranker
        self._candidate_n = candidate_n

    def ranked_indices(self, query: str) -> list[int]:
        candidates = self._base.ranked_indices(query)[: self._candidate_n]
        scores = self._reranker.score(query, [self.chunks[i].text for i in candidates])
        order = sorted(range(len(candidates)), key=lambda j: -scores[j])
        return [candidates[j] for j in order]
