"""Shared retrieval/generation/guard/log path for demo and future evaluation.

The frozen v1 experiments remain untouched. This is a new execution version;
its live quality and latency must be measured before promoting old scores.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import threading
import time
import uuid
from pathlib import Path

from src.artifacts import fingerprint
from src.evaluation.paid_calls_v1 import CallConfig
from src.evaluation.prompts_v1 import generation_messages, validate_generation
from src.generation.pii import redact_payload
from src.generation.prompt import detect_language_hint
from src.retrieval.candidates_v1 import CANDIDATES


@dataclasses.dataclass(frozen=True)
class ServiceConfig:
    candidate: str = "baseline_dense_k10"
    confidence_threshold: float = 0.55
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    max_history_messages: int = 40
    max_question_characters: int = 8000
    max_history_characters: int = 40000

    def __post_init__(self):
        if self.candidate not in CANDIDATES or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Unsupported candidate or similarity threshold")
        if not 0 <= self.temperature <= 2 or min(self.max_history_messages, self.max_question_characters, self.max_history_characters) < 1:
            raise ValueError("Invalid temperature or input limit")
        self.call_config()

    def call_config(self):
        return CallConfig(model=self.model, temperature=self.temperature, max_tokens=self.max_output_tokens)


def validate_input(question, history, config):
    if not isinstance(question, str) or not question.strip() or len(question) > config.max_question_characters:
        raise ValueError("Question must be nonempty and within the configured limit")
    if not isinstance(history, list) or len(history) > config.max_history_messages:
        raise ValueError("Conversation history exceeds the configured limit; reset the session")
    for turn in history:
        if not isinstance(turn, dict) or set(turn) != {"role", "utterance"} or turn["role"] not in {"user", "agent"} or not isinstance(turn["utterance"], str):
            raise ValueError("Invalid role-labelled history")
    if sum(len(t["utterance"]) for t in history) > config.max_history_characters:
        raise ValueError("Conversation history exceeds the configured character budget; reset the session")


def localized(question, en, zh):
    return zh if detect_language_hint(question).startswith("Chinese") else en


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(redact_payload(event), ensure_ascii=False, allow_nan=False) + "\n"
    # Restrictive permissions for new files; existing user's log is not chmod-ed.
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        stream.write(serialized)


class RagRuntime:
    def __init__(self, retriever, caller, config=None, log_path=None, artifact_binding=None):
        self.retriever, self.caller = retriever, caller
        self.config = config or ServiceConfig()
        self.log_path = log_path
        self.run_id = uuid.uuid4().hex
        self.binding = {"runtime_version": "full-corpus-service-1", "config": dataclasses.asdict(self.config),
                        "artifacts": artifact_binding or {"status": "injected_test_dependencies"}}
        self.config_sha256 = fingerprint(self.binding)
        # A single queue covers model access and ledger writes. Report the wait
        # time instead of silently excluding it from request latency.
        self.lock = threading.RLock()

    def answer(self, question, history, session_id, turn_id):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        row = {"request_id": request_id, "session_id": session_id, "turn_id": turn_id, "run_id": self.run_id,
            "runtime_version": self.binding["runtime_version"], "config_sha256": self.config_sha256,
            "status": "failed", "response": None, "contexts": [], "retrieval_contexts": None,
            "citations": [], "citation_check": "not_evaluated", "error_type": None,
            "call": None, "latency_ms": {}, "generation_cache_hit": False}
        with self.lock:
            row["latency_ms"]["queue_wait"] = (time.perf_counter() - started) * 1000
            try:
                validate_input(question, history, self.config)
                stage = time.perf_counter()
                item = {"question": question, "conversation_history": history}
                chunks, retrieval = self.retriever.retrieve(item, self.config.candidate)
                row["latency_ms"]["retrieval"] = (time.perf_counter() - stage) * 1000
                row["retrieval"] = {k: v for k, v in retrieval.items() if k != "query"}
                row["retrieval_contexts"] = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]
                if not chunks or retrieval["top1_dense_score"] < self.config.confidence_threshold:
                    reason = "no_retrieval_results" if not chunks else "low_retrieval_confidence"
                    row.update(status="ok", refusal_reason=reason, citation_check="not_applicable",
                        response={"answer": localized(question,
                            "I don't have enough relevant information in this knowledge base. Please clarify the question or ask about the covered services.",
                            "知识库中没有足够的相关信息。请补充问题细节，或询问知识库覆盖的服务。"), "action": "refuse", "citations": []})
                else:
                    row["contexts"] = list(row["retrieval_contexts"])
                    stage = time.perf_counter()
                    call = self.caller.call("service:generation:" + self.config_sha256,
                        generation_messages(item, chunks), self.config.call_config(), validate_generation)
                    row["latency_ms"]["generation"] = (time.perf_counter() - stage) * 1000
                    row.update(call=call, generation_cache_hit=bool(call.get("cache_hit")))
                    if call["status"] != "ok":
                        row.update(status=call["status"], error_type="generation_failed")
                    else:
                        response = call["parsed"]
                        validate_generation(response)
                        valid = {c.chunk_id: c for c in chunks}
                        invalid = [cid for cid in response["citations"] if cid not in valid]
                        row["raw_citations"] = response["citations"]
                        if invalid or (response["action"] == "answer" and not response["citations"]):
                            row.update(status="failed", error_type="invalid_or_missing_citations", citation_check="failed")
                        else:
                            row.update(status="ok", response=response, citation_check="valid_ids_only_not_semantic_support")
                            for cid in dict.fromkeys(response["citations"]):
                                chunk = valid[cid]
                                row["citations"].append({"chunk_id": cid, "document": chunk.doc_id,
                                    "heading": list(getattr(chunk, "heading_path", [])),
                                    "source_file": Path(getattr(chunk, "source_file", "")).name,
                                    "excerpt": chunk.text})
            except ValueError as exc:
                row.update(error_type=type(exc).__name__, status="failed")
            except Exception as exc:
                # Never copy SDK exception strings: they may contain payloads
                # or credentials. Full raw completions are not logged here.
                row.update(error_type=type(exc).__name__, status="failed")
            row["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
            if self.log_path:
                attempts = (row.get("call") or {}).get("attempts", [])
                event = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    **{k: row[k] for k in ("request_id", "session_id", "turn_id", "run_id", "runtime_version", "config_sha256",
                                         "status", "citation_check", "error_type", "latency_ms", "generation_cache_hit")},
                    "query": question if isinstance(question, str) else "[invalid question type]",
                    "answer": (row["response"] or {}).get("answer"),
                    "retrieved_chunk_ids": [c["chunk_id"] for c in row["retrieval_contexts"] or []],
                    "citations": (row["response"] or {}).get("citations", []), "raw_citations": row.get("raw_citations", []),
                    "model": self.config.model, "attempts": attempts,
                    "usage_scope": "On cache hits, attempt details describe the original provider call, not new billing."}
                try:
                    append_event(self.log_path, event)
                    row["logging_status"] = "ok"
                except OSError:
                    row["logging_status"] = "failed"
            row["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
        return row


class Session:
    """In-memory single conversation; reset clears prior text and rotates ID."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.session_id = uuid.uuid4().hex
        self.history = []
        self.turn_id = 0

    def ask(self, runtime, question):
        self.turn_id += 1
        result = runtime.answer(question, list(self.history), self.session_id, self.turn_id)
        if result["status"] == "ok":
            self.history.extend([{"role": "user", "utterance": question}, {"role": "agent", "utterance": result["response"]["answer"]}])
        return result
