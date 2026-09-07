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

# Unicode word boundaries treat Chinese letters and adjacent digits as one
# word. Digit/ASCII boundaries allow e.g. 手机号13800138000 to be redacted.
# Match complete IDs before phone patterns to avoid redacting only a prefix.
_PATTERNS = [
    (re.compile(r"(?<![0-9A-Za-z])[0-9]{17}[0-9Xx](?![0-9A-Za-z])"), "[ID_NUMBER]"),
    (re.compile(r"(?<![0-9A-Za-z])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9A-Za-z])"), "[SSN]"),
    (re.compile(r"(?<![A-Za-z0-9_.+\-])[A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"), "[EMAIL]"),
    (re.compile(r"(?<![0-9A-Za-z])(?:\+?86[-. \t]?)?1[3-9][0-9](?:[-. \t]?[0-9]){8}(?![0-9A-Za-z])"), "[PHONE]"),
    (re.compile(r"(?<![0-9A-Za-z])(?:\+?1[-. \t]?)?(?:\([0-9]{3}\)|[0-9]{3})[-. \t]?[0-9]{3}[-. \t]?[0-9]{4}(?![0-9A-Za-z])"), "[PHONE]"),
    (re.compile(r"(?<![0-9A-Za-z])[0-9]{9,}(?![0-9A-Za-z])"), "[ID_NUMBER]"),
]


def redact_pii(text: str) -> str:
    if not text:
        return text
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


_SECRET_KEYS = {"authorization", "proxy_authorization", "api_key", "apikey", "deepseek_api_key",
                "openai_api_key", "password", "secret", "access_token", "refresh_token"}


def redact_payload(value):
    """Redact JSON-like log values recursively; schema keys stay stable.

    This does not change model inputs or stored source documents. Pattern
    coverage is deliberately limited; it is not comprehensive PII detection.
    """
    if isinstance(value, dict):
        return {key: "[REDACTED]" if str(key).lower().replace("-", "_") in _SECRET_KEYS
                else redact_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_pii(value)
    return value
