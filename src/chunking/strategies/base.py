from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    domain: str
    language: str
    source_file: str
    strategy: str
    chunk_size_config: int
    overlap_config: int
    heading_path: list[str]
    text: str
    token_count: int
    # Evaluation-only provenance; never appended to embedding/generator text.
    source_spans: list[dict] = field(default_factory=list)


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}::c{index:04d}"
