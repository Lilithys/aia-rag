"""Shared recursive-splitting and greedy-packing primitives.

Both the recursive-structural (B) and semantic-section (C) strategies need
to cut an oversized piece of text down to a token budget without cutting
mid-word, and both need to merge small pieces back up to fill a budget.
These two operations are implemented once here.
"""
from __future__ import annotations

from .tokenizer import count_tokens, decode, encode, tail_text

# Largest-to-smallest separators to prefer when a piece of text must be cut:
# paragraph breaks, then line breaks, then sentence ends (CJK and Latin), then
# clause breaks, then whitespace. "" as a last resort means "hard token cut".
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " ", ""]


def smart_join(pieces: list[str]) -> str:
    """Joins text pieces the way a human would: a blank line between pieces
    unless that would visually separate what is really one CJK-run/sentence
    continuation (rare here since pieces are paragraph/sentence-sized)."""
    return "\n\n".join(p for p in (piece.strip() for piece in pieces) if p)


def _hard_token_split(text: str, max_tokens: int) -> list[str]:
    ids = encode(text)
    if not ids:
        return []
    out = []
    for i in range(0, len(ids), max_tokens):
        out.append(decode(ids[i : i + max_tokens]))
    return out


def _merge_to_budget(parts: list[str], sep: str, max_tokens: int) -> list[str]:
    """Greedily re-joins consecutive parts (with sep) up to max_tokens each."""
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for part in parts:
        if not part:
            continue
        part_tokens = count_tokens(part)
        if current and current_tokens + part_tokens > max_tokens:
            groups.append(sep.join(current))
            current, current_tokens = [], 0
        current.append(part)
        current_tokens += part_tokens
    if current:
        groups.append(sep.join(current))
    return groups


def split_to_size(text: str, max_tokens: int) -> list[str]:
    """Cuts text into pieces each within max_tokens, preferring the largest
    separator that actually reduces piece size, recursing into any piece
    still too big. Falls back to a hard token-boundary cut."""
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    for sep in _SEPARATORS:
        if sep == "":
            return _hard_token_split(text, max_tokens)
        if sep in text:
            parts = [p for p in text.split(sep) if p.strip()]
            if len(parts) <= 1:
                continue
            grouped = _merge_to_budget(parts, sep, max_tokens)
            if len(grouped) <= 1:
                continue  # this separator didn't actually split anything usefully
            out: list[str] = []
            for g in grouped:
                out.extend(split_to_size(g, max_tokens))
            return out
    return _hard_token_split(text, max_tokens)


def apply_overlap(pieces: list[str], overlap_tokens: int) -> list[str]:
    """Prepends the tail of each piece onto the next, so consecutive chunks
    share `overlap_tokens` of context."""
    if overlap_tokens <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        carry = tail_text(prev, overlap_tokens)
        out.append(smart_join([carry, cur]) if carry else cur)
    return out


def greedy_pack(units: list[str], max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Packs a sequence of text units (each assumed <= max_tokens on its own)
    into groups as close to max_tokens as possible without exceeding it,
    carrying `overlap_tokens` of trailing context from one chunk into the
    next."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for u in units:
        u = u.strip()
        if not u:
            continue
        u_tokens = count_tokens(u)
        if current and current_tokens + u_tokens > max_tokens:
            chunks.append(smart_join(current))
            if overlap_tokens > 0:
                carry = tail_text(chunks[-1], overlap_tokens)
                current = [carry] if carry else []
                current_tokens = count_tokens(carry) if carry else 0
            else:
                current, current_tokens = [], 0
        current.append(u)
        current_tokens += u_tokens
    if current:
        chunks.append(smart_join(current))
    return chunks
