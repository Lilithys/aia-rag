"""Shared helpers: frontmatter rendering and CJK-aware text joining."""
from __future__ import annotations

import re
import unicodedata

import yaml


def is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0xFF00 <= cp <= 0xFFEF  # Fullwidth forms / punctuation
        or 0x3000 <= cp <= 0x303F  # CJK punctuation
    )


def smart_join(tokens: list[str]) -> str:
    """Join word tokens the way a human would: no space around CJK runs,
    single space between Latin/number words."""
    out = ""
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if not out:
            out = tok
            continue
        if is_cjk_char(out[-1]) or is_cjk_char(tok[0]):
            out += tok
        else:
            out += " " + tok
    return out


def make_frontmatter(meta: dict) -> str:
    ordered_keys = [
        "doc_id",
        "domain",
        "title",
        "language",
        "format",
        "source_file",
        "conversion_method",
    ]
    ordered = {k: meta[k] for k in ordered_keys if k in meta and meta[k] is not None}
    for k, v in meta.items():
        if k not in ordered:
            ordered[k] = v
    body = yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{body}---\n\n"


def normalize_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def word_count(text: str) -> int:
    """Rough word count that treats each CJK character as one word."""
    cjk = sum(1 for ch in text if is_cjk_char(ch))
    non_cjk_text = "".join(" " if is_cjk_char(ch) else ch for ch in text)
    latin_words = len(non_cjk_text.split())
    return cjk + latin_words
