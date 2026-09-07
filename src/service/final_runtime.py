"""Explicit-folder runtime factory for the delivery configuration."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from src.artifacts import file_sha256
from src.evaluation.paid_calls_v1 import BudgetLedger, JsonCaller, atomic_json
from src.retrieval.folder_retriever import FolderRetriever
from src.service.runtime import RagRuntime, ServiceConfig


def preflight(folder: Path, config: ServiceConfig) -> dict:
    folder = Path(folder)
    chunks = folder / "chunks.jsonl"
    embedding_manifest = folder / "embeddings/bge_m3/embedding_manifest.json"
    embedding_array = folder / "embeddings/bge_m3/chunk_embeddings.npy"
    for path in (chunks, embedding_manifest, embedding_array):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(embedding_manifest.read_text())
    if metadata.get("embedding_sha256") != file_sha256(embedding_array):
        raise ValueError("Embedding array differs from its manifest")
    return {
        "config": dataclasses.asdict(config),
        "artifact_folder": str(folder),
        "chunks_sha256": file_sha256(chunks),
        "embedding_manifest_sha256": file_sha256(embedding_manifest),
        "embedding_sha256": file_sha256(embedding_array),
        "encoder": metadata["encoder"],
        "chunk_count": len(metadata["chunks"]),
        "api_endpoint": "https://api.deepseek.com/chat/completions",
        "thinking": "disabled",
        "code_sha256": {str(path): file_sha256(path) for path in (
            Path(__file__), Path("src/service/runtime.py"), Path("src/retrieval/folder_retriever.py"),
            Path("src/retrieval/candidates_v1.py"), Path("src/retrieval/retriever.py"),
            Path("src/retrieval/indexes.py"), Path("src/retrieval/chroma_store.py"), Path("src/evaluation/prompts_v1.py"),
            Path("src/evaluation/paid_calls_v1.py"), Path("src/generation/pii.py"),
        )},
    }


def make_runtime(folder: Path, state_dir: Path, config: ServiceConfig, budget_rmb: float):
    if not math.isfinite(budget_rmb) or budget_rmb <= 0:
        raise ValueError("Budget must be a finite positive number")
    binding = preflight(folder, config)
    ledger = BudgetLedger(state_dir / "usage.jsonl", budget_rmb)
    caller = JsonCaller(state_dir / "api_cache", ledger)
    retriever = FolderRetriever(folder)
    runtime = RagRuntime(retriever, caller, config, state_dir / "requests.jsonl", binding)
    atomic_json(state_dir / "runs" / (runtime.run_id + ".json"), {
        "run_id": runtime.run_id,
        "config_sha256": runtime.config_sha256,
        "binding": runtime.binding,
    })
    return runtime, ledger
