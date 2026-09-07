"""Persistent derived index, checked against chunk content and vector payloads.

HNSW agreement with exact cosine ranking is measured separately in the
versioned full-corpus experiment; persistence does not imply exact search.
"""
from __future__ import annotations

import chromadb
import numpy as np
from src.artifacts import chunk_inputs, fingerprint
import dataclasses

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
        "record_sha256": fingerprint(dataclasses.asdict(chunk)),
    }


def get_collection(
    chunks: list[ChunkRecord],
    embeddings,
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    chunk_inputs(chunks)  # Reject duplicate IDs before any database mutation.
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks) or not np.isfinite(embeddings).all():
        raise ValueError("invalid embeddings for Chroma collection")
    client = chromadb.PersistentClient(path=persist_dir, settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    expected_ids = [c.chunk_id for c in chunks]
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    existing = stored["ids"]
    if existing and len(stored["embeddings"][0]) != embeddings.shape[1]:
        # A Chroma collection retains its dimension even after deleting rows.
        # Only this derived collection is rebuilt; source/chunk artifacts remain.
        client.delete_collection(collection_name)
        collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        existing = []
    matches = set(existing) == set(expected_ids)
    if matches:
        positions = {cid: i for i, cid in enumerate(existing)}
        for i, chunk in enumerate(chunks):
            j = positions[chunk.chunk_id]
            if (stored["documents"][j] != chunk.text
                or stored["metadatas"][j] != _metadata(chunk)
                or not np.array_equal(np.asarray(stored["embeddings"][j], dtype=np.float32), embeddings[i])):
                matches = False
                break
    if not matches:
        if existing:
            collection.delete(ids=existing)
        collection.add(
            ids=expected_ids,
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )
    return collection
