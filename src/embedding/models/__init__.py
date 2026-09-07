from .base import EmbeddingModel
from .bge_m3 import BgeM3Model

REGISTRY = {
    "bge_m3": BgeM3Model,
}

__all__ = ["EmbeddingModel", "BgeM3Model", "REGISTRY"]
