"""Structured JSONL logging for generation monitoring and issue diagnosis
(require.md: "design a logging system suitable for generation monitoring
and issue diagnosis" + "PII-redacted sample logs"). One line per query,
enough to answer both "is this fast/cheap enough" (latency, token usage)
and "why did this answer look wrong" (retrieval scores, refusal reason,
citation validity) without needing to reproduce the query.

`session_id` links every call within one multi-turn conversation, so a
diagnosis question ("why did turn 3 of this conversation fail?") can pull
every prior turn's log line instead of just the one that broke. It defaults
to `request_id` when a caller doesn't have (or doesn't need) a real session
concept -- a single isolated call is its own one-line session.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field

from .pii import redact_pii

LOG_PATH = "data/logs/generation.jsonl"


@dataclass
class LogEntry:
    timestamp: float
    request_id: str
    session_id: str
    query_redacted: str
    retrieval_top1_score: float
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    refused: bool
    refusal_reason: str | None
    answer_redacted: str | None
    citations: list[str]
    citations_valid: bool
    latency_ms: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)
    model: str | None = None


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_interaction(entry: LogEntry, path: str = LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def redact_entry_text(query: str, answer: str | None) -> tuple[str, str | None]:
    return redact_pii(query), redact_pii(answer) if answer is not None else None
