"""Strategy A: fixed-size chunking (structure-blind baseline).

Slides a fixed-width token window across the whole document body with a
fixed token overlap. Ignores headings, paragraphs and sentences entirely --
a chunk boundary can fall in the middle of a word or sentence. This is the
naive baseline every other strategy is compared against.

heading_path is still attached to each chunk (as metadata, not injected into
the chunk text) by locating which section the chunk's start offset falls
in -- useful for citation display and for the evidence-integrity eval, but
it plays no role in where the boundaries are.
"""
from __future__ import annotations

from ..doc_tree import Node, build_tree, flatten_sections, heading_path, preorder
from ..tokenizer import count_tokens, decode, encode
from .base import ChunkRecord, make_chunk_id

NAME = "fixed_size"


def _section_offsets(body: str, tree: Node) -> list[tuple[int, list[str]]]:
    """Returns [(char_offset_of_section_start, heading_path)], sorted by
    offset, by locating each section's heading line in the body text."""
    offsets = []
    search_from = 0
    for node in preorder(tree):
        marker = f"{'#' * node.level} {node.heading}"
        idx = body.find(marker, search_from)
        if idx == -1:
            idx = body.find(marker)
        if idx != -1:
            offsets.append((idx, heading_path(node)))
            search_from = idx + len(marker)
    offsets.sort(key=lambda x: x[0])
    return offsets


def _heading_path_at(offset: int, section_offsets: list[tuple[int, list[str]]]) -> list[str]:
    path: list[str] = []
    for start, p in section_offsets:
        if start <= offset:
            path = p
        else:
            break
    return path


def chunk_document(
    body: str, meta: dict, chunk_size: int, overlap: int, **_
) -> list[ChunkRecord]:
    body = body.strip()
    if not body:
        return []

    tree = build_tree(flatten_sections(body))
    section_offsets = _section_offsets(body, tree)

    ids = encode(body)
    step = max(chunk_size - overlap, 1)
    records: list[ChunkRecord] = []
    index = 0
    start_tok = 0
    while start_tok < len(ids):
        end_tok = min(start_tok + chunk_size, len(ids))
        window_ids = ids[start_tok:end_tok]
        text = decode(window_ids).strip()
        if text:
            char_offset = len(decode(ids[:start_tok]))
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(meta["doc_id"], index),
                    doc_id=meta["doc_id"],
                    domain=meta["domain"],
                    language=meta["language"],
                    source_file=meta["source_file"],
                    strategy=NAME,
                    chunk_size_config=chunk_size,
                    overlap_config=overlap,
                    heading_path=_heading_path_at(char_offset, section_offsets),
                    text=text,
                    token_count=count_tokens(text),
                )
            )
            index += 1
        if end_tok == len(ids):
            break
        start_tok += step
    return records
