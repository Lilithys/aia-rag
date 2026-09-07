"""Content fingerprints shared by embedding caches and persistent indexes."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_inputs(chunks) -> list[dict]:
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk IDs")
    return [{"chunk_id": c.chunk_id, "text_sha256": hashlib.sha256(c.text.encode()).hexdigest()} for c in chunks]


def chunk_records_fingerprint(chunks) -> str:
    return fingerprint([dataclasses.asdict(c) for c in chunks])
