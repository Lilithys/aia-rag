"""Production default: dense-only Chroma + BGE-M3.

This default was selected on the legacy corpus. The frozen full-corpus dev
comparison is documented in doc/evaluation/v1/retrieval_review.md; windowed
reranking is a candidate pending generation/latency validation.

Usage:
    from src.retrieval.retriever import get_default_retriever
    retriever = get_default_retriever()
    chunks = retriever.search("some question", top_k=10)
"""
from __future__ import annotations

from pathlib import Path

from src.chunking.strategies.base import ChunkRecord
from src.embedding.embed_corpus import embed_chunks_cached, load_chunks, load_verified_embeddings
from src.embedding.models.bge_m3 import BgeM3Model

from .chroma_store import get_collection
from .indexes import ChromaDenseIndex

DEFAULT_TOP_K = 10  # retained baseline; tune with the versioned full-corpus dev protocol


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
    persist_dir: str | None = None,
) -> Retriever:
    chunks = load_chunks(chunks_path)
    model = BgeM3Model()
    try:
        embeddings = load_verified_embeddings(chunks, embeddings_path, expected_encoder=model.cache_identity())
    except (FileNotFoundError, ValueError) as exc:
        path = Path(embeddings_path)
        if path.name == "chunk_embeddings.npy" and path.parent.name == model.name:
            # Preserve existing demo entrypoints: migrate old ID-only local
            # caches once, or recompute changed rows. Never bless unverified rows.
            embeddings = embed_chunks_cached(model, chunks, str(path.parent.parent))
        else:
            raise ValueError(
                "Unverified/stale embeddings at a nonstandard path: rebuild with "
                "python -m src.embedding.embed_corpus --chunks <chunks path> "
                "--cache-dir <embedding cache root> and use its output path"
            ) from exc
    collection = get_collection(chunks, embeddings, **({"persist_dir": persist_dir} if persist_dir else {}))
    return Retriever(ChromaDenseIndex(chunks, collection, model))
