"""Common interface every embedding backend implements.

Each model has its own convention for how queries and passages should be
prepared before encoding (e5 needs "query: "/"passage: " prefixes, OpenAI
needs neither, BGE-M3 needs neither) -- that goes in `prepare_query` /
`prepare_passage`, not left to the caller, so every backend is used the way
its own docs say it should be, and encode_* always takes raw text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    name: str
    dim: int
    is_local: bool  # local model (compute cost) vs hosted API (dollar cost)

    @abstractmethod
    def encode_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized embeddings."""

    @abstractmethod
    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized embeddings."""
