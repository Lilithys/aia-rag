"""Loads the document manifest that lists every source document to convert."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class DocEntry:
    doc_id: str
    domain: str
    title: str
    language: str
    format: str
    file: str  # path relative to doc_root, e.g. "documents/markdown/dmv_001.md"


def load_manifest(doc_root: str) -> list[DocEntry]:
    manifest_path = os.path.join(doc_root, "manifests", "document_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for d in data["documents"]:
        entries.append(
            DocEntry(
                doc_id=d["doc_id"],
                domain=d["domain"],
                title=d["title"],
                language=d["language"],
                format=d["format"],
                file=d["file"],
            )
        )
    return entries
