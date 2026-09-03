"""OpenAI text-embedding-3-large -- hosted API, 3072-dim, no prefix convention.

Not run in the current comparison (no API key configured in this
environment) but implemented so it drops into the same pipeline once a key
is available: set OPENAI_API_KEY and pass --models openai to run_experiment.
"""
from __future__ import annotations

import os

import numpy as np

from .base import EmbeddingModel

MODEL_ID = "text-embedding-3-large"
# https://openai.com/api/pricing/ as of this writing
PRICE_PER_1M_TOKENS_USD = 0.13


class OpenAIEmbeddingModel(EmbeddingModel):
    name = "openai"
    dim = 3072
    is_local = False

    def __init__(self):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self._client = OpenAI(api_key=api_key)
        self.total_tokens_used = 0

    def _embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=MODEL_ID, input=batch)
            self.total_tokens_used += resp.usage.total_tokens
            out.extend(d.embedding for d in resp.data)
        return np.array(out, dtype="float32")

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._embed(texts, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._embed(texts, batch_size)

    def estimated_cost_usd(self) -> float:
        return self.total_tokens_used / 1_000_000 * PRICE_PER_1M_TOKENS_USD
