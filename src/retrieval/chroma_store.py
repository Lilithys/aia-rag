"""Persistent Chroma collection for the BGE-M3 chunk embeddings.

199 chunks doesn't need a server or an ANN index -- Chroma's own default
(HNSW) is functionally brute-force at this scale -- but it does give real
on-disk persistence and metadata filtering (domain, language, doc_id) for
free, which is what actually matters for a "real" retriever versus the
ephemeral numpy arrays used for the step-2/3 comparisons.
"""
from __future__ import annotations

import chromadb

from src.chunking.strategies.base import ChunkRecord

COLLECTION_NAME = "chunks_bge_m3"
PERSIST_DIR = "data/processed/chroma_db"


def _metadata(chunk: ChunkRecord) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "domain": chunk.domain,
        "language": chunk.language,
        "source_file": chunk.source_file,
        "heading_path": " > ".join(chunk.heading_path),
        "token_count": chunk.token_count,
    }


def get_collection(
    chunks: list[ChunkRecord],
    embeddings,
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    expected_ids = [c.chunk_id for c in chunks]
    existing = collection.get(include=[])["ids"]
    if set(existing) != set(expected_ids):
        if existing:
            collection.delete(ids=existing)
        collection.add(
            ids=expected_ids,
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )
    return collection
