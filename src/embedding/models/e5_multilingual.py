"""intfloat/multilingual-e5-large -- local, 1024-dim.

e5 models are trained with an explicit "query: " / "passage: " prefix
convention and score notably worse without it -- this is documented behavior,
not an optional nicety, so it's applied here rather than left to the caller.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel

MODEL_ID = "intfloat/multilingual-e5-large"


class E5MultilingualModel(EmbeddingModel):
    name = "multilingual_e5"
    dim = 1024
    is_local = True

    def __init__(self, device: str | None = None):
        self._model = SentenceTransformer(MODEL_ID, device=device)

    def encode_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        return self._model.encode(
            prefixed, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return self._model.encode(
            prefixed, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
