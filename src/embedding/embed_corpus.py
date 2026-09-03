"""Embeds the materialized chunk set with one model, caching to disk so
re-running the comparison doesn't re-embed (encoding is the slow part for
local models)."""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from src.chunking.strategies.base import ChunkRecord

from .models.base import EmbeddingModel


def load_chunks(path: str = "data/processed/chunks/chunks.jsonl") -> list[ChunkRecord]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(ChunkRecord(**json.loads(line)))
    return chunks


def embed_chunks_cached(
    model: EmbeddingModel, chunks: list[ChunkRecord], cache_dir: str = "data/processed/embeddings"
) -> np.ndarray:
    model_dir = os.path.join(cache_dir, model.name)
    os.makedirs(model_dir, exist_ok=True)
    emb_path = os.path.join(model_dir, "chunk_embeddings.npy")
    ids_path = os.path.join(model_dir, "chunk_ids.json")

    expected_ids = [c.chunk_id for c in chunks]
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        cached_ids = json.load(open(ids_path, encoding="utf-8"))
        if cached_ids == expected_ids:
            return np.load(emb_path)

    embeddings = model.encode_passages([c.text for c in chunks])
    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(expected_ids, f)
    return embeddings
