"""Converts image-only (scanned) PDFs to Markdown via OCR.

There is no text layer and no font metadata to lean on, so headings are
recovered from *visual* salience instead:

  - Tesseract (`eng`, `chi_sim`, or both -- picked from the document's known
    language) gives us each OCR'd line's bounding box.
  - We additionally sample the mean ink color inside that box from the
    rendered page image. A line is a "styled" (heading-candidate) line if
    its color noticeably departs from grayscale (chroma = max(RGB)-min(RGB))
    -- this catches headings that are colored but not enlarged, which plain
    font-size comparison would miss.
  - A line is also heading-candidate if it is visibly taller than the
    page's typical body-text line.
  - Among heading-candidate lines, height relative to the single tallest
    heading line on the document sorts them into 3 tiers (H1/H2/H3).

This is a best-effort heuristic (documented limitation): OCR cannot recover
the exact original heading depth, only relative visual prominence. Plain
grayscale scans (no color cues at all) still work, degrading gracefully to
size-only classification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymupdf
import pytesseract
from PIL import Image

from .common import is_cjk_char, smart_join

DPI = 300
# Tesseract is more accurate given the true language than the combined pack --
# mixing in chi_sim for a pure-English page (or vice versa) measurably increases
# misreads. Only documents actually mixing scripts need the combined pack.
LANG_BY_DOC_LANGUAGE = {
    "en": "eng",
    "zh": "chi_sim",
    "mixed": "chi_sim+eng",
}
DEFAULT_TESS_LANG = "chi_sim+eng"
CHROMA_THRESHOLD = 12  # max(R,G,B)-min(R,G,B) beyond which a line counts as "colored"
SIZE_HEADING_RATIO = 1.15  # body-relative height ratio that alone implies a heading
TIER1_RATIO = 0.85  # relative to the tallest heading line on the doc
TIER2_RATIO = 0.55
LIST_PREFIXES = ("-", "*", "•")


@dataclass
class Line:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    height: int
    color: tuple[float, float, float]

    @property
    def chroma(self) -> float:
        return max(self.color) - min(self.color)


def _ocr_lines(image: Image.Image, tess_lang: str) -> list[Line]:
    arr = np.asarray(image.convert("RGB"))
    data = pytesseract.image_to_data(image, lang=tess_lang, output_type=pytesseract.Output.DICT)
    grouped: dict[tuple[int, int, int], dict] = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        l, t, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        g = grouped.setdefault(key, {"tokens": [], "l": [], "t": [], "r": [], "b": []})
        g["tokens"].append(text)
        g["l"].append(l)
        g["t"].append(t)
        g["r"].append(l + w)
        g["b"].append(t + h)

    lines: list[Line] = []
    for g in grouped.values():
        x0, y0, x1, y1 = min(g["l"]), min(g["t"]), max(g["r"]), max(g["b"])
        text = smart_join(g["tokens"])
        if not text:
            continue
        band = arr[y0:y1, x0:x1]
        if band.size == 0:
            color = (0.0, 0.0, 0.0)
        else:
            gray = band.mean(axis=2)
            mask = gray < 210
            color = tuple(band[mask].mean(axis=0)) if mask.any() else (0.0, 0.0, 0.0)
        lines.append(Line(text=text, x0=x0, y0=y0, x1=x1, y1=y1, height=y1 - y0, color=color))

    lines.sort(key=lambda ln: (ln.y0, ln.x0))
    return lines


def _body_median_height(lines: list[Line]) -> float:
    heights = sorted(ln.height for ln in lines)
    if not heights:
        return 1.0
    return heights[len(heights) // 2]


def _is_list_line(text: str) -> bool:
    return text.startswith(LIST_PREFIXES)


def _classify(lines: list[Line]) -> list[tuple[Line, int]]:
    """Returns (line, level) pairs; level 0 means "body text"."""
    if not lines:
        return []
    body_h = _body_median_height(lines)

    candidates = [
        ln for ln in lines if ln.chroma >= CHROMA_THRESHOLD or ln.height >= body_h * SIZE_HEADING_RATIO
    ]
    candidate_ids = {id(ln) for ln in candidates}
    max_heading_h = max((ln.height for ln in candidates), default=body_h)

    out: list[tuple[Line, int]] = []
    for ln in lines:
        is_candidate = id(ln) in candidate_ids
        if not is_candidate or _is_list_line(ln.text):
            out.append((ln, 0))
            continue
        ratio = ln.height / max_heading_h if max_heading_h else 0
        if ratio >= TIER1_RATIO:
            out.append((ln, 1))
        elif ratio >= TIER2_RATIO:
            out.append((ln, 2))
        else:
            out.append((ln, 3))
    return out


def _merge_adjacent_headings(classified: list[tuple[Line, int]]) -> list[tuple[Line, int]]:
    """Wrapped multi-line headings (close vertically, similar height) become one heading,
    even if a slight OCR height wobble put them a tier apart."""
    merged: list[tuple[Line, int]] = []
    for ln, level in classified:
        if level and merged and merged[-1][1]:
            prev_ln, prev_level = merged[-1]
            gap = ln.y0 - prev_ln.y1
            height_ratio = min(ln.height, prev_ln.height) / max(ln.height, prev_ln.height)
            if gap < prev_ln.height * 0.6 and height_ratio > 0.7:
                joined_text = smart_join([prev_ln.text, ln.text])
                combined = Line(
                    text=joined_text,
                    x0=min(prev_ln.x0, ln.x0),
                    y0=prev_ln.y0,
                    x1=max(prev_ln.x1, ln.x1),
                    y1=ln.y1,
                    height=max(prev_ln.height, ln.height),
                    color=prev_ln.color,
                )
                merged[-1] = (combined, min(level, prev_level))
                continue
        merged.append((ln, level))
    return merged


def _assemble_markdown(classified: list[tuple[Line, int]]) -> str:
    out: list[str] = []
    block_lines: list[Line] = []

    def flush_block():
        if not block_lines:
            return
        text = smart_join([l.text for l in block_lines])
        out.append(text)
        block_lines.clear()

    prev_bottom = None
    body_h = _body_median_height([ln for ln, lvl in classified if lvl == 0]) or 1.0
    for ln, level in classified:
        if level:
            flush_block()
            out.append(f"{'#' * level} {ln.text}")
            prev_bottom = ln.y1
            continue

        starts_new_list_item = _is_list_line(ln.text)
        big_gap = prev_bottom is not None and (ln.y0 - prev_bottom) > body_h * 1.6

        # A new list item, or a large vertical gap, starts a fresh block. Otherwise
        # the line is a wrapped continuation of whatever block is currently open
        # (a paragraph, or the previous list item's text running onto a new line).
        if starts_new_list_item or big_gap:
            flush_block()
        block_lines.append(ln)
        prev_bottom = ln.y1

    flush_block()
    return "\n\n".join(out)


def convert(path: str, language: str | None = None) -> str:
    tess_lang = LANG_BY_DOC_LANGUAGE.get(language, DEFAULT_TESS_LANG)
    doc = pymupdf.open(path)
    page_markdowns = []
    for page in doc:
        pix = page.get_pixmap(dpi=DPI)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        lines = _ocr_lines(image, tess_lang)
        classified = _classify(lines)
        classified = _merge_adjacent_headings(classified)
        md = _assemble_markdown(classified)
        if md.strip():
            page_markdowns.append(md.strip())
    doc.close()
    return ("\n\n".join(page_markdowns)).strip() + "\n"
