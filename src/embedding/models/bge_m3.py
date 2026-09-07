"""BAAI/bge-m3 -- local, multilingual, 1024-dim.

Unlike earlier BGE versions, bge-m3's own model card says no instruction
prefix is needed for either queries or passages -- that's one of its stated
design goals, so passages and queries are encoded identically here.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel

MODEL_ID = "BAAI/bge-m3"


class BgeM3Model(EmbeddingModel):
    name = "bge_m3"
    dim = 1024
    is_local = True

    def __init__(self, device: str | None = None, revision: str | None = None):
        self.revision = revision
        self._model = SentenceTransformer(MODEL_ID, device=device, revision=revision)

    def cache_identity(self) -> dict:
        return self.sentence_transformer_identity(MODEL_ID, "", "")

    def encode_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.encode_passages(texts, batch_size=batch_size)
