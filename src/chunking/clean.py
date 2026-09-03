"""Cleans converted Markdown documents before chunking.

Two independent problems, found by inspecting the step-1 output
(`data/processed/markdown/`):

  1. PDF text extraction (pymupdf4llm) backslash-escapes literal punctuation
     that happened to look like Markdown syntax in the source PDF text
     (`\\[`, `\\]`, `\\$`, `\\*`). We don't render these files as Markdown
     HTML anywhere -- they're chunked as plain text for embedding/LLM
     ingestion -- so the escaping is just noise. Unescape it.

  2. ~8 documents (both Chinese and English) end with leaked CMS/UI field
     labels from the original scraped source -- e.g. "Show \"do it online\"
     button in megamenu:", "在大菜单中显示"在线办理"按钮:", "Related PDFs:"
     followed by bare filenames, "yes or no survey:". These are backend
     toggles, never answerable content. Whole heading sections matching
     this denylist are dropped.
"""
from __future__ import annotations

import re
import unicodedata

from .doc_tree import FlatSection, flatten_sections

_ESCAPE_RE = re.compile(r"\\([\[\]\$\*])")

_BOILERPLATE_HEADING_SUBSTRINGS = [
    "菜单中显示",
    "megamenu中显示",
    "是否禁用此交易",
    "是或否调查",
    "相关pdf文件",
    "button in megamenu",
    "disable this transaction",
    "yes or no survey",
    "related pdfs",
]


def _is_boilerplate_heading(heading: str) -> bool:
    low = heading.lower()
    return any(s in low for s in _BOILERPLATE_HEADING_SUBSTRINGS)


def _unescape(text: str) -> str:
    return _ESCAPE_RE.sub(r"\1", text)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")  # non-breaking space
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_body(body: str) -> tuple[str, dict]:
    """Returns (cleaned_body, stats) where stats records what was removed."""
    sections = flatten_sections(body)
    kept: list[FlatSection] = []
    dropped = 0
    for s in sections:
        if s.heading and _is_boilerplate_heading(s.heading):
            dropped += 1
            continue
        kept.append(s)

    escape_fixes = len(_ESCAPE_RE.findall(body))

    out: list[str] = []
    for s in kept:
        text = _normalize_text(_unescape(s.text))
        if s.heading is not None:
            out.append(f"{'#' * s.level} {_normalize_text(_unescape(s.heading))}")
        if text:
            out.append(text)

    cleaned = "\n\n".join(out).strip()
    if cleaned:
        cleaned += "\n"
    stats = {"boilerplate_sections_dropped": dropped, "escape_sequences_fixed": escape_fixes}
    return cleaned, stats
