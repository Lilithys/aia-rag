"""Converts .docx documents to Markdown using paragraph styles.

Word heading styles ("Title"/"Document Title", "Heading 1".."Heading 9")
map directly to Markdown '#'..'######', which is far more reliable than any
visual heuristic. Document title -> H1, "Heading N" -> H(N+1), capped at 6.
"""
from __future__ import annotations

import re

import docx
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

_HEADING_RE = re.compile(r"^Heading\s*(\d+)$", re.IGNORECASE)


def _iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("unsupported parent type")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _heading_level(style_name: str) -> int | None:
    name = (style_name or "").strip()
    if name in ("Title", "Document Title"):
        return 1
    m = _HEADING_RE.match(name)
    if m:
        return min(int(m.group(1)) + 1, 6)
    return None


def _list_prefix(paragraph: Paragraph) -> tuple[str | None, int]:
    """Returns (marker, indent_level) for list paragraphs, else (None, 0)."""
    style = (paragraph.style.name or "") if paragraph.style else ""
    m = re.match(r"^List (Bullet|Number)\s*(\d*)$", style, re.IGNORECASE)
    if not m:
        return None, 0
    kind = m.group(1).lower()
    level = int(m.group(2)) if m.group(2) else 1
    marker = "-" if kind == "bullet" else "1."
    return marker, level - 1


def _paragraph_text(paragraph: Paragraph) -> str:
    parts = []
    for run in paragraph.runs:
        text = run.text
        if not text:
            continue
        if run.bold and run.italic:
            text = f"***{text}***"
        elif run.bold:
            text = f"**{text}**"
        elif run.italic:
            text = f"*{text}*"
        parts.append(text)
    return "".join(parts).strip()


def _table_to_markdown(table: Table) -> str:
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def convert(path: str) -> str:
    document = docx.Document(path)
    out: list[str] = []
    list_counters: dict[int, int] = {}

    for block in _iter_block_items(document):
        if isinstance(block, Table):
            list_counters.clear()
            md_table = _table_to_markdown(block)
            if md_table:
                out.append(md_table)
            continue

        paragraph = block
        text = _paragraph_text(paragraph)
        if not text:
            continue

        level = _heading_level(paragraph.style.name if paragraph.style else "")
        if level is not None:
            list_counters.clear()
            out.append(f"{'#' * level} {text}")
            continue

        marker, indent = _list_prefix(paragraph)
        if marker == "-":
            list_counters.clear()
            out.append(f"{'  ' * indent}- {text}")
            continue
        if marker == "1.":
            list_counters[indent] = list_counters.get(indent, 0) + 1
            out.append(f"{'  ' * indent}{list_counters[indent]}. {text}")
            continue

        list_counters.clear()
        out.append(text)

    return "\n\n".join(out).strip() + "\n"
