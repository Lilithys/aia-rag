"""Loads the cleaned Markdown corpus for chunking."""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import yaml

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)", re.DOTALL)


@dataclass
class CleanDoc:
    meta: dict  # doc_id, domain, title, language, format, source_file, conversion_method
    body: str


def load_clean_corpus(clean_dir: str = "data/interim/markdown_clean") -> list[CleanDoc]:
    docs = []
    for path in sorted(glob.glob(os.path.join(clean_dir, "*.md"))):
        text = open(path, encoding="utf-8").read()
        m = _FRONTMATTER_RE.match(text)
        if not m:
            continue
        meta = yaml.safe_load(m.group(1))
        docs.append(CleanDoc(meta=meta, body=m.group(2).strip()))
    return docs
