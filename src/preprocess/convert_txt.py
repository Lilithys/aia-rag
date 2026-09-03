"""Converts plain .txt documents to Markdown.

Plain text has no markup at all, so headings must be inferred. We use a
simple, documented heuristic:

  - The first non-blank line is always the document title (H1).
  - Any other line that is alone in its paragraph block, and does not end
    like a normal sentence (no trailing '.', '。', '!', '!', ',', '，', ';'),
    is treated as a section heading (H2). Plain text carries no reliable
    signal for deeper nesting, so all detected sub-headings come out flat
    at H2 -- this is a known limitation of txt-to-Markdown recovery.
  - Lines starting with '-', '*', '•' or "<number>. " are kept as Markdown
    list items.
  - Everything else is a body paragraph.
"""
from __future__ import annotations

import re

from .common import is_cjk_char, smart_join

_SENTENCE_ENDERS = tuple(".。!!,，;；")
_LEAKED_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")
_LIST_PREFIXES = ("- ", "* ", "• ")


def _is_list_line(line: str) -> bool:
    if line.startswith(_LIST_PREFIXES):
        return True
    # "1. " / "1) " style numbered list items
    head = line.split(" ", 1)[0]
    return head.rstrip(".)").isdigit() and len(head) <= 4 and " " in line


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _is_list_line(stripped):
        return False
    if stripped.endswith(_SENTENCE_ENDERS):
        return False
    max_len = 60 if any(is_cjk_char(c) for c in stripped) else 100
    return len(stripped) <= max_len


def _split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _join_block(lines: list[str]) -> str:
    if any(is_cjk_char(c) for line in lines for c in line):
        return smart_join(lines)
    return " ".join(lines)


def convert(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = _split_blocks(raw)
    out: list[str] = []
    title_used = False

    for block in blocks:
        if not title_used:
            out.append(f"# {block[0].strip()}")
            rest = block[1:]
            title_used = True
            if not rest:
                continue
            block = rest

        if len(block) == 1:
            leaked = _LEAKED_HEADING_RE.match(block[0])
            if leaked:
                out.append(f"{leaked.group(1)} {leaked.group(2)}")
                continue
            if _looks_like_heading(block[0]):
                out.append(f"## {block[0]}")
                continue

        if all(_is_list_line(l) for l in block):
            for l in block:
                out.append(l)
            continue

        out.append(_join_block(block))

    return "\n\n".join(out).strip() + "\n"
