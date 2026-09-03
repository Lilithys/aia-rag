from .base import EmbeddingModel
from .bge_m3 import BgeM3Model
from .e5_multilingual import E5MultilingualModel
from .openai_embed import OpenAIEmbeddingModel

REGISTRY = {
    "bge_m3": BgeM3Model,
    "multilingual_e5": E5MultilingualModel,
    "openai": OpenAIEmbeddingModel,
}

__all__ = ["EmbeddingModel", "BgeM3Model", "E5MultilingualModel", "OpenAIEmbeddingModel", "REGISTRY"]
