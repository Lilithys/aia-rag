"""Passthrough converter for documents that are already Markdown.

Strips whatever frontmatter the source file has (so we can replace it with
our own normalized schema) and keeps the body -- headings and all -- as-is.
"""
from __future__ import annotations

import re

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def convert(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _FRONTMATTER_RE.sub("", text, count=1)
    return text.strip() + "\n"
