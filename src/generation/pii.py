"""Basic PII redaction for logs -- require.md asks for "basic PII handling"
and PII-redacted sample logs specifically. This runs only at the logging
boundary: the LLM still sees the real query (it needs to, to answer), this
just keeps sensitive values out of persisted logs.

Deliberately simple pattern matching, not a full PII-detection model --
scoped to the categories a DMV/SSA/VA/StudentAid QA bot will actually see:
SSNs, emails, phone numbers, and long all-digit ID-like numbers.
"""
from __future__ import annotations

import re

_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{9,}\b"), "[ID_NUMBER]"),  # SSNs w/o dashes, DMV/case IDs, etc.
]


def redact_pii(text: str) -> str:
    if not text:
        return text
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text
