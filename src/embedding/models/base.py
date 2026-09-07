"""Common interface every embedding backend implements.

Each model has its own convention for how queries and passages should be
prepared before encoding (e5 needs "query: "/"passage: " prefixes, OpenAI
needs neither, BGE-M3 needs neither) -- that goes in `prepare_query` /
`prepare_passage`, not left to the caller, so every backend is used the way
its own docs say it should be, and encode_* always takes raw text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import inspect
from importlib.metadata import version

import numpy as np


class EmbeddingModel(ABC):
    name: str
    dim: int
    is_local: bool  # local model (compute cost) vs hosted API (dollar cost)

    def cache_identity(self) -> dict:
        """Encoder behavior, never credentials or client/environment state."""
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "dimension": self.dim,
            "implementation_sha256": hashlib.sha256((inspect.getsource(type(self)) + inspect.getsource(EmbeddingModel)).encode()).hexdigest(),
        }

    def sentence_transformer_identity(self, model_id: str, passage_prefix: str, query_prefix: str) -> dict:
        model = self._model
        transformer = model._first_module()
        revision = getattr(transformer.auto_model.config, "_commit_hash", None) or getattr(self, "revision", None)
        if not revision:
            raise ValueError("Cannot cache local embeddings without a resolved model revision")
        return {
            **self.cache_identity_base(),
            "model_id": model_id, "revision": revision,
            "max_seq_length": model.max_seq_length,
            "passage_prefix": passage_prefix, "query_prefix": query_prefix,
            "normalize_embeddings": True,
            "device": str(model.device),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "sentence_transformers_version": version("sentence-transformers"),
            "transformers_version": version("transformers"),
            "torch_version": version("torch"),
            "modules": [
                {"class": type(module).__name__, "config": module.get_config_dict() if hasattr(module, "get_config_dict") else {}}
                for module in model.children()
            ],
        }

    def cache_identity_base(self) -> dict:
        return EmbeddingModel.cache_identity(self)

    @abstractmethod
    def encode_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized embeddings."""

    @abstractmethod
    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized embeddings."""
