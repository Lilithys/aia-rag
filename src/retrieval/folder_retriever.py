"""Run the frozen v1 candidate rankings against an explicit artifact folder."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.retrieval.candidates_v1 import CandidateRetriever, RERANK_CONFIG


class FolderRetriever(CandidateRetriever):
    """CandidateRetriever whose folder is explicit instead of convention based."""

    def __init__(self, folder: Path):
        import torch
        from sentence_transformers import CrossEncoder

        from src.embedding.embed_corpus import load_chunks, load_verified_embeddings
        from src.embedding.models.bge_m3 import BgeM3Model
        from src.retrieval.chroma_store import get_collection
        from src.retrieval.indexes import ChromaDenseIndex
        from src.retrieval.retriever import Retriever

        folder = Path(folder)
        torch.set_num_threads(4)
        started = time.perf_counter()
        chunks = load_chunks(str(folder / "chunks.jsonl"))
        embedding_path = folder / "embeddings/bge_m3/chunk_embeddings.npy"
        manifest = json.loads((embedding_path.parent / "embedding_manifest.json").read_text())
        revision = manifest["encoder"].get("revision")
        if not revision:
            raise ValueError("Embedding manifest does not pin the encoder revision")
        # Query encoding must use the same device identity that was committed
        # with the passage vectors; otherwise a strict cache check would fall
        # through into an expensive, unplanned full-corpus re-embedding.
        model = BgeM3Model(device=manifest["encoder"]["device"], revision=revision)
        embeddings = load_verified_embeddings(chunks, embedding_path, expected_encoder=model.cache_identity())
        collection = get_collection(chunks, embeddings, persist_dir=str(folder / "chroma"))
        self.dense = Retriever(
            ChromaDenseIndex(chunks, collection, model)
        )
        self.reranker = CrossEncoder(
            RERANK_CONFIG["model"],
            revision=RERANK_CONFIG["revision"],
            local_files_only=True,
            processor_kwargs={"model_max_length": 512},
        )
        self.reranker.predict([["query", "passage"]], show_progress_bar=False)
        self.dense.search("user: What services are available?", top_k=20)
        self.setup_seconds = time.perf_counter() - started
