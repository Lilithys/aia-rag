"""Embeds the materialized chunk set with one model, caching to disk so
re-running the comparison doesn't re-embed (encoding is the slow part for
local models)."""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from src.chunking.strategies.base import ChunkRecord
from src.artifacts import chunk_inputs, file_sha256, fingerprint

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
    inputs = chunk_inputs(chunks)
    identity = model.cache_identity()
    input_fingerprint = fingerprint({"chunks": inputs, "encoder": identity})
    try:
        return load_verified_embeddings(chunks, emb_path, expected_encoder=identity)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        pass  # Old ID-only caches are deliberately rebuilt, never certified retroactively.

    embeddings = np.empty((len(chunks), model.dim), dtype=np.float32)
    reusable = {}
    old_values = None
    try:
        old_manifest, old_values = _read_cache_payload(emb_path, expected_encoder=identity)
        reusable = {row["text_sha256"]: i for i, row in enumerate(old_manifest["chunks"])}
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        pass
    missing = []
    for i, row in enumerate(inputs):
        previous = reusable.get(row["text_sha256"])
        if previous is None:
            missing.append(i)
        else:
            embeddings[i] = old_values[previous]
    started = time.perf_counter()
    if missing:
        encoded = model.encode_passages([chunks[i].text for i in missing])
        if encoded.shape != (len(missing), model.dim) or not np.isfinite(encoded).all():
            raise ValueError("encoder returned invalid embedding shape or nonfinite values")
        embeddings[missing] = encoded
    encode_seconds = time.perf_counter() - started
    if embeddings.shape != (len(chunks), model.dim) or not np.isfinite(embeddings).all():
        raise ValueError("encoder returned invalid embedding shape or nonfinite values")
    manifest_path = Path(model_dir) / "embedding_manifest.json"
    # Invalidate the old commit marker before replacing payloads. A interrupted
    # or concurrent write can be rejected by the file hashes, never silently used.
    manifest_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=".embedding-", dir=model_dir) as tmp:
        root = Path(tmp)
        np.save(root / "chunk_embeddings.npy", embeddings)
        (root / "chunk_ids.json").write_text(json.dumps(expected_ids, ensure_ascii=False) + "\n")
        manifest = {
            "schema_version": 1, "input_fingerprint": input_fingerprint,
            "encoder": identity, "chunks": inputs,
            "embedding_shape": list(embeddings.shape), "embedding_dtype": str(embeddings.dtype),
            "embedding_sha256": file_sha256(root / "chunk_embeddings.npy"),
            "ids_sha256": file_sha256(root / "chunk_ids.json"),
            "encoded_rows": len(missing), "reused_rows": len(chunks) - len(missing),
            "encode_seconds": encode_seconds,
        }
        (root / "embedding_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for name in ("chunk_embeddings.npy", "chunk_ids.json", "embedding_manifest.json"):
            os.replace(root / name, Path(model_dir) / name)
    return embeddings


def load_verified_embeddings(chunks, embeddings_path, expected_encoder: dict | None = None) -> np.ndarray:
    manifest, result = _read_cache_payload(embeddings_path, expected_encoder)
    if manifest["chunks"] != chunk_inputs(chunks):
        raise ValueError("embedding cache does not match ordered chunk content")
    return result


def _read_cache_payload(embeddings_path, expected_encoder: dict | None = None):
    path = Path(embeddings_path)
    manifest = json.loads((path.parent / "embedding_manifest.json").read_text())
    inputs = manifest["chunks"]
    if manifest["schema_version"] != 1 or len({c["chunk_id"] for c in inputs}) != len(inputs):
        raise ValueError("invalid cache schema or duplicate chunk IDs")
    encoder = manifest["encoder"]
    if expected_encoder is not None and encoder != expected_encoder:
        raise ValueError("embedding encoder configuration changed")
    if fingerprint({"chunks": inputs, "encoder": encoder}) != manifest["input_fingerprint"]:
        raise ValueError("invalid embedding input fingerprint")
    if file_sha256(path) != manifest["embedding_sha256"] or file_sha256(path.parent / "chunk_ids.json") != manifest["ids_sha256"]:
        raise ValueError("embedding cache payload hash mismatch")
    if json.loads((path.parent / "chunk_ids.json").read_text()) != [c["chunk_id"] for c in inputs]:
        raise ValueError("embedding IDs/order mismatch")
    result = np.load(path, allow_pickle=False)
    if list(result.shape) != manifest["embedding_shape"] or result.shape != (len(inputs), encoder["dimension"]):
        raise ValueError("embedding shape mismatch")
    if str(result.dtype) != manifest["embedding_dtype"] or not np.isfinite(result).all():
        raise ValueError("invalid embedding values/dtype")
    return manifest, result


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build content-verified local BGE-M3 embeddings; no evaluation or paid API")
    parser.add_argument("--chunks", default="data/processed/chunks/chunks.jsonl")
    parser.add_argument("--cache-dir", default="data/processed/embeddings")
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .models.bge_m3 import BgeM3Model
    model = BgeM3Model(revision=args.revision)
    result = embed_chunks_cached(model, load_chunks(args.chunks), args.cache_dir)
    print(f"Verified embeddings: {result.shape}; cache={args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
