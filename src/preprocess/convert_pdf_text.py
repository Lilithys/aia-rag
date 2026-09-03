"""Converts text-layer PDFs to Markdown via pymupdf4llm.

pymupdf4llm reconstructs heading levels from font-size/boldness heuristics
across the extracted text layer, which is the standard approach for
digitally-native PDFs (no OCR needed since the text is already there).
"""
from __future__ import annotations

import contextlib
import io

import pymupdf4llm

from .common import normalize_blank_lines


def convert(path: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        md = pymupdf4llm.to_markdown(path, show_progress=False)
    return normalize_blank_lines(md)
