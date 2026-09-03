"""Parses a cleaned Markdown document body into its heading structure.

Two views of the same structure are exposed:

  - `flatten_sections`: a flat, in-order list of (heading, own body text)
    pairs. Used by the cleaner (drop-a-section-by-heading-match is trivial
    on a flat list) and as the basis for the tree.
  - `build_tree` / `Node`: the nested heading hierarchy, needed by the
    structural chunkers to know a section's full ancestor path and to
    reason about "this section plus everything nested under it".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tokenizer import count_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class FlatSection:
    level: int  # 0 = preamble before any heading (rare/empty in this corpus)
    heading: str | None
    text: str  # this section's own body text, not including any subsections


def flatten_sections(body: str) -> list[FlatSection]:
    lines = body.split("\n")
    sections: list[FlatSection] = []
    level = 0
    heading: str | None = None
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip("\n").strip()
        if heading is not None or text:
            sections.append(FlatSection(level=level, heading=heading, text=text))

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return sections


@dataclass
class Node:
    level: int
    heading: str | None
    text: str
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None


def build_tree(sections: list[FlatSection]) -> Node:
    root = Node(level=0, heading=None, text="")
    stack: list[Node] = [root]
    for s in sections:
        if s.heading is None:
            root.text = s.text
            continue
        node = Node(level=s.level, heading=s.heading, text=s.text)
        while stack[-1].level >= node.level:
            stack.pop()
        node.parent = stack[-1]
        stack[-1].children.append(node)
        stack.append(node)
    return root


def heading_path(node: Node) -> list[str]:
    path = []
    n = node
    while n is not None and n.parent is not None:
        path.append(n.heading)
        n = n.parent
    return list(reversed(path))


def preorder(node: Node):
    for c in node.children:
        yield c
        yield from preorder(c)


def subtree_token_count(node: Node) -> int:
    total = count_tokens(node.text)
    for c in node.children:
        total += count_tokens(f"{'#' * c.level} {c.heading}") + subtree_token_count(c)
    return total


def render_subtree(node: Node) -> str:
    parts = []
    if node.text.strip():
        parts.append(node.text.strip())
    for c in node.children:
        parts.append(f"{'#' * c.level} {c.heading}")
        rendered = render_subtree(c)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(p for p in parts if p.strip())
