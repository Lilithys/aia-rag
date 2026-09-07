"""Token counting shared by every chunking strategy.

We size chunks in tokens (via tiktoken's cl100k_base) rather than characters
or words: it is embedding/LLM-context-budget-relevant, and gives a single
consistent unit across English, Chinese and mixed text where a "word count"
means very different things per language. cl100k_base is used as a
model-agnostic proxy for chunk sizing -- the eventual embedding/LLM choice
is a separate decision made later in the project.
"""
from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def encode(text: str) -> list[int]:
    return _ENCODING.encode(text)


def decode(tokens: list[int], errors: str = "replace") -> str:
    return _ENCODING.decode(tokens, errors=errors)


def tail_text(text: str, n_tokens: int) -> str:
    """Returns the last n_tokens worth of text, on a token boundary."""
    if n_tokens <= 0:
        return ""
    ids = _ENCODING.encode(text)
    start = max(0, len(ids) - n_tokens)
    while start < len(ids):
        try:
            return _ENCODING.decode(ids[start:], errors="strict")
        except UnicodeDecodeError:
            start += 1  # omit an incomplete leading character from overlap only
    return ""
