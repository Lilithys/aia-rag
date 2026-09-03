"""Evidence-matching and BM25 retrieval helpers used by the chunking
comparison. Kept separate from evaluate.py so evaluate.py stays focused on
the metric definitions.
"""
from __future__ import annotations

import difflib
import re

import jieba
from rank_bm25 import BM25Okapi

_WS_RE = re.compile(r"\s+")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


def normalize_for_match(s: str) -> str:
    return _WS_RE.sub("", s)


def evidence_found_in_chunk(evidence: str, chunk_text: str) -> bool:
    """True if `evidence` appears verbatim in `chunk_text` (whitespace-
    insensitive), or, for longer spans, if a close (>=85% covered) fuzzy
    match exists -- OCR'd/extracted text can differ from the benchmark's
    gold text by a handful of misread characters without the underlying
    content being genuinely missing."""
    ev = normalize_for_match(evidence)
    ch = normalize_for_match(chunk_text)
    if not ev:
        return False
    if ev in ch:
        return True
    if len(ev) < 8:
        return False
    matcher = difflib.SequenceMatcher(None, ch, ev, autojunk=False)
    covered = sum(b.size for b in matcher.get_matching_blocks())
    return covered / len(ev) >= 0.85


def bm25_tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok or _PUNCT_ONLY_RE.match(tok):
            continue
        tokens.append(tok)
    return tokens


class Bm25Index:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self._bm25 = BM25Okapi([bm25_tokenize(c.text) for c in chunks]) if chunks else None

    def ranked_indices(self, query: str) -> list[int]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(bm25_tokenize(query))
        return sorted(range(len(scores)), key=lambda i: -scores[i])
