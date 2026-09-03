"""Strategy C: semantic / section-aware chunking.

Goal: preserve whole heading sections as the retrieval unit, not a
consistent token size. `chunk_size` here is a *soft merge threshold*: a
section plus everything nested under it becomes exactly one chunk as long
as it fits within chunk_size * MERGE_RATIO. Small subsections are pulled
up into their parent's chunk rather than fragmented; a section's own lead-in
text (before its first subheading) rides along with the first child's chunk
rather than being dropped or split off on its own. Only a section that is
still oversized even alone falls back to the same recursive splitter
strategy B uses (with overlap applied between the resulting pieces).

The distinguishing move versus strategy B: every chunk is prefixed with its
full heading breadcrumb *inside the chunk text itself* (not just metadata),
e.g. "[Doc Title > Section > Subsection]\\n<content>" -- a well-established
technique for hierarchical documents, since the embedding then reflects the
section's context even for a chunk whose own text is generic ("Step 1: ...").
"""
from __future__ import annotations

from ..doc_tree import Node, build_tree, flatten_sections, heading_path, render_subtree, subtree_token_count
from ..splitting import apply_overlap, smart_join, split_to_size
from ..tokenizer import count_tokens
from .base import ChunkRecord, make_chunk_id

NAME = "semantic_section"
MERGE_RATIO = 1.5  # a section (+ descendants) up to this multiple of chunk_size stays whole


def _breadcrumb(path: list[str]) -> str:
    return " > ".join(path)


def _prefixed(path: list[str], text: str) -> str:
    bc = _breadcrumb(path)
    return f"[{bc}]\n{text}" if bc else text


def chunk_document(body: str, meta: dict, chunk_size: int, overlap: int, **_) -> list[ChunkRecord]:
    body = body.strip()
    if not body:
        return []
    root = build_tree(flatten_sections(body))

    emitted: list[tuple[list[str], str]] = []  # (heading_path, full chunk text incl. breadcrumb)

    def visit(node: Node, inherited_lead: str):
        subtotal = subtree_token_count(node) + count_tokens(inherited_lead)
        keep_whole = (not node.children) or (subtotal <= chunk_size * MERGE_RATIO)

        if keep_whole:
            rendered = render_subtree(node)
            text = smart_join([inherited_lead, rendered]) if inherited_lead else rendered
            if not text.strip():
                return
            path = heading_path(node)
            full = _prefixed(path, text)
            if count_tokens(full) > chunk_size * 2:
                # pathologically large leaf section: fall back to recursive splitting
                pieces = apply_overlap(split_to_size(text, chunk_size), overlap)
                for p in pieces:
                    emitted.append((path, _prefixed(path, p)))
            else:
                emitted.append((path, full))
            return

        # too big to keep whole: split at child-heading boundaries. This
        # node's own lead-in text (before its first subheading) rides along
        # with the first child rather than being dropped or split off alone.
        own = node.text.strip()
        for i, child in enumerate(node.children):
            visit(child, own if i == 0 else "")

    visit(root, "")

    records: list[ChunkRecord] = []
    for i, (path, text) in enumerate(emitted):
        records.append(
            ChunkRecord(
                chunk_id=make_chunk_id(meta["doc_id"], i),
                doc_id=meta["doc_id"],
                domain=meta["domain"],
                language=meta["language"],
                source_file=meta["source_file"],
                strategy=NAME,
                chunk_size_config=chunk_size,
                overlap_config=overlap,
                heading_path=path,
                text=text,
                token_count=count_tokens(text),
            )
        )
    return records
