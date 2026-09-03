"""Strategy B: recursive structural chunking.

Goal: consistent chunk sizes close to `chunk_size`, using structure
(headings > paragraphs > sentences) as the *preferred* cut points instead of
arbitrary token boundaries.

  1. Walk sections in document order. Any section whose own text exceeds
     chunk_size gets recursively split at the largest separator available
     (paragraph, then sentence, ...) via `splitting.split_to_size`.
  2. The resulting sequence of (heading_path, piece) units -- now each
     individually within budget -- is greedily packed back together across
     section boundaries up to chunk_size, so small sections merge with
     their neighbors instead of becoming undersized chunks.
  3. Overlap is carried between consecutive emitted chunks.

heading_path metadata is carried per chunk (the path of whichever section
contributed the first unit) but, unlike strategy C, is not injected into the
chunk text.
"""
from __future__ import annotations

from ..doc_tree import build_tree, flatten_sections, heading_path, preorder
from ..splitting import smart_join, split_to_size
from ..tokenizer import count_tokens, tail_text
from .base import ChunkRecord, make_chunk_id

NAME = "recursive_structural"


def chunk_document(body: str, meta: dict, chunk_size: int, overlap: int, **_) -> list[ChunkRecord]:
    body = body.strip()
    if not body:
        return []
    tree = build_tree(flatten_sections(body))

    units: list[tuple[list[str], str]] = []  # (heading_path, text)
    for node in preorder(tree):
        # The heading line itself is real, embeddable content (a section
        # title can directly answer "what covers X"), not just navigation --
        # so it rides along as the first line of the section's first piece,
        # same as a reader would see it, rather than living only in metadata.
        heading_line = f"{'#' * node.level} {node.heading}"
        if not node.text.strip():
            units.append((heading_path(node), heading_line))
            continue
        pieces = split_to_size(node.text, chunk_size)
        for i, piece in enumerate(pieces):
            text = f"{heading_line}\n{piece}" if i == 0 else piece
            units.append((heading_path(node), text))
    if not units:
        return []

    records: list[ChunkRecord] = []
    index = 0
    current_texts: list[str] = []
    current_path: list[str] | None = None
    current_tokens = 0

    def flush():
        nonlocal index, current_texts, current_path, current_tokens
        if not current_texts:
            return
        text = smart_join(current_texts)
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
                heading_path=current_path or [],
                text=text,
                token_count=count_tokens(text),
            )
        )
        index += 1

    for path, piece in units:
        piece_tokens = count_tokens(piece)
        if current_texts and current_tokens + piece_tokens > chunk_size:
            flush()
            carry = tail_text(records[-1].text, overlap) if overlap > 0 and records else ""
            current_texts = [carry] if carry else []
            current_tokens = count_tokens(carry) if carry else 0
            current_path = None
        if current_path is None:
            current_path = path
        current_texts.append(piece)
        current_tokens += piece_tokens
    flush()
    return records
