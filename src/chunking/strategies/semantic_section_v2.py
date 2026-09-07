"""Semantic-section chunking v2: preserve heading-only leaf content.

OCR can place an entire sentence or paragraph on a Markdown heading line.
When a large parent is recursively visited, v1 skips a leaf whose body is
empty before its heading can become an embedding breadcrumb. This version
emits that heading as content. The rule does not inspect evaluation data.
"""
from __future__ import annotations

from ..doc_tree import Node, build_tree, flatten_sections, heading_path, render_subtree, subtree_token_count
from ..splitting import apply_overlap, smart_join, split_to_size
from ..tokenizer import count_tokens
from .base import ChunkRecord, make_chunk_id

NAME = "semantic_section_v2"
MERGE_RATIO = 1.5


def _prefixed(path: list[str], text: str) -> str:
    breadcrumb = " > ".join(path)
    return f"[{breadcrumb}]\n{text}" if breadcrumb else text


def chunk_document(body: str, meta: dict, chunk_size: int, overlap: int, **_) -> list[ChunkRecord]:
    body = body.strip()
    if not body:
        return []
    root = build_tree(flatten_sections(body))
    emitted: list[tuple[list[str], str]] = []

    def emit(path: list[str], text: str):
        full = _prefixed(path, text)
        if count_tokens(full) > chunk_size * 2:
            for piece in apply_overlap(split_to_size(text, chunk_size), overlap):
                emitted.append((path, _prefixed(path, piece)))
        else:
            emitted.append((path, full))

    def visit(node: Node, inherited_lead: str):
        subtotal = subtree_token_count(node) + count_tokens(inherited_lead)
        keep_whole = (not node.children) or subtotal <= chunk_size * MERGE_RATIO
        if keep_whole:
            rendered = render_subtree(node)
            text = smart_join([inherited_lead, rendered]) if inherited_lead else rendered
            if text.strip():
                emit(heading_path(node), text)
            elif node.heading:
                # The node heading is the body; ancestors remain a breadcrumb.
                emitted.append((heading_path(node), _prefixed(heading_path(node)[:-1], node.heading)))
            return
        own = smart_join([inherited_lead, node.text])
        for index, child in enumerate(node.children):
            visit(child, own if index == 0 else "")

    visit(root, "")
    return [ChunkRecord(chunk_id=make_chunk_id(meta["doc_id"], index), doc_id=meta["doc_id"],
                        domain=meta["domain"], language=meta["language"], source_file=meta["source_file"],
                        strategy=NAME, chunk_size_config=chunk_size, overlap_config=overlap,
                        heading_path=path, text=text, token_count=count_tokens(text))
            for index, (path, text) in enumerate(emitted)]
