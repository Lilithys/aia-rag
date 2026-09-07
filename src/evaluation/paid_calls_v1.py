"""Explicit DeepSeek JSON calls with audited retries, caching and a shared CNY cap.

No credentials or clients are created at import time. The CLI holds a process
lock; the ledger also serializes its concurrent judge workers. Incomplete
reservations survive a crash and count at their upper bound until reconciled.
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
from zoneinfo import ZoneInfo

from src.artifacts import fingerprint

PRICING = {
    "checked_on": "2026-09-06",
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "currency": "CNY", "unit": "per million tokens",
    "peak": {"deepseek-v4-flash": [0.10, 3.0, 9.0], "deepseek-v4-pro": [0.30, 9.0, 27.0]},
    "columns": ["cached_input", "uncached_input", "output"],
    "off_peak_multiplier": 0.5,
    "peak_schedule": "Asia/Shanghai weekdays [09:00,12:00) and [14:00,18:00)",
    "documented_versions": {"deepseek-v4-flash": "DeepSeek-V4-Flash-0731", "deepseek-v4-pro": "DeepSeek-V4-Pro-0813"},
    "version_limitation": "API names are mutable aliases; response model/id/fingerprint are recorded when returned.",
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclasses.dataclass(frozen=True)
class CallConfig:
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 45
    max_attempts: int = 2
    input_token_upper_limit: int = 65536
    thinking: str = "disabled"

    def __post_init__(self):
        if self.model not in PRICING["peak"] or self.thinking != "disabled":
            raise ValueError("Only priced DeepSeek models with thinking disabled are supported")
        if self.max_attempts not in (1, 2) or not 1 <= self.max_tokens <= 2048:
            raise ValueError("This smoke runner permits at most two attempts and 2048 output tokens")
        if self.timeout_seconds <= 0 or self.input_token_upper_limit <= 0:
            raise ValueError("Timeout and input limit must be positive")


def input_upper_bound(messages) -> int:
    # Conservative byte-token bound plus chat framing margin; cl100k is used
    # only for forecasts, never as the provider's measured token count.
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024


def price(model, prompt_tokens, completion_tokens, cached_tokens=0, multiplier=1.0):
    if not 0 <= cached_tokens <= prompt_tokens or min(prompt_tokens, completion_tokens) < 0:
        raise ValueError("Invalid usage counts")
    cached, uncached, output = PRICING["peak"][model]
    return ((prompt_tokens - cached_tokens) * uncached + cached_tokens * cached + completion_tokens * output) / 1e6 * multiplier


def is_peak(timestamp: float) -> bool:
    local = dt.datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Shanghai"))
    hour = local.hour + local.minute / 60
    return local.weekday() < 5 and (9 <= hour < 12 or 14 <= hour < 18)


def interval_multiplier(start, end):
    # Requests crossing an hour boundary use the maximum applicable rate.
    samples = [start, end] + list(range(int(start // 3600 + 1) * 3600, int(end), 3600))
    return 1.0 if any(is_peak(t) for t in samples) else 0.5


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: Path, cap_rmb: float):
        if cap_rmb <= 0:
            raise ValueError("Budget must be positive")
        self.path, self.cap = path, cap_rmb
        self.lock = threading.RLock()
        self.events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def receipts(self):
        reservations = {e["attempt_id"]: dict(e) for e in self.events if e["event"] == "reserve"}
        for event in self.events:
            if event["event"] == "settle":
                reservations[event["attempt_id"]].update(event)
        return list(reservations.values())

    def spent_upper(self):
        return sum(r.get("accounted_rmb", r["reserved_rmb"]) for r in self.receipts())

    def reserve(self, role, config, request_hash, tokens):
        amount = price(config.model, tokens, config.max_tokens)
        with self.lock:
            if self.spent_upper() + amount > self.cap + 1e-12:
                raise BudgetExceeded("Cumulative billed/reserved estimate would exceed the configured cap")
            attempt = uuid.uuid4().hex
            self._append({"event": "reserve", "attempt_id": attempt, "role": role,
                          "model": config.model, "request_hash": request_hash,
                          "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "reserved_rmb": amount, "input_token_upper_bound": tokens,
                          "max_output_tokens": config.max_tokens})
            return attempt, amount

    def settle(self, attempt_id, reserved, usage, config, start, end, outcome):
        actual = None
        if usage is not None:
            actual = price(config.model, usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"], interval_multiplier(start, end))
            accounted = actual
        else:
            accounted = reserved  # Unknown provider billing is never silently zeroed.
        event = {"event": "settle", "attempt_id": attempt_id, "outcome": outcome,
                 "usage": usage, "usage_cost_estimate_rmb": actual, "accounted_rmb": accounted,
                 "usage_unknown": usage is None, "seconds": end - start}
        with self.lock:
            self._append(event)
        return event


def extract_usage(response):
    raw = getattr(response, "usage", None)
    if raw is None:
        return None
    raw = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    prompt, completion = raw.get("prompt_tokens"), raw.get("completion_tokens")
    details = raw.get("prompt_tokens_details") or {}
    cached = raw.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0)) or 0
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (prompt, completion, cached)):
        return None
    if not 0 <= cached <= prompt or completion < 0:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached,
            "reasoning_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0), "raw": raw}


class JsonCaller:
    def __init__(self, cache_root: Path, ledger: BudgetLedger, client=None):
        self.cache_root, self.ledger = cache_root, ledger
        self._key_locks = {}
        self._locks_guard = threading.Lock()
        if client is None:
            from openai import OpenAI
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY; configure the environment or use --env-file")
            client = OpenAI(api_key=key, base_url="https://api.deepseek.com", max_retries=0)
        self.client = client

    def call(self, role, messages, config: CallConfig, validate):
        request = {"schema_version": 1, "role": role, "messages": messages, "config": dataclasses.asdict(config)}
        request_hash = fingerprint(request)
        with self._locks_guard:
            lock = self._key_locks.setdefault(request_hash, threading.Lock())
        with lock:
            return self._call_locked(request, request_hash, role, messages, config, validate)

    def _call_locked(self, request, request_hash, role, messages, config, validate):
        cache = self.cache_root / (request_hash + ".json")
        if cache.exists():
            envelope = json.loads(cache.read_text())
            if envelope.get("sha256") != fingerprint(envelope.get("payload")):
                raise ValueError("Corrupt API cache; inspect before repeating a paid request")
            payload = envelope["payload"]
            if payload["request"] != request:
                raise ValueError("API cache input mismatch")
            validate(payload["result"]["parsed"])
            return {**payload["result"], "cache_hit": True}
        bound = input_upper_bound(messages)
        if bound > config.input_token_upper_limit:
            return {"status": "input_budget_exceeded", "cache_hit": False, "attempts": [], "request_hash": request_hash}
        attempts = []
        for index in range(config.max_attempts):
            try:
                attempt_id, reserved = self.ledger.reserve(role, config, request_hash, bound)
            except BudgetExceeded:
                return {"status": "budget_exceeded", "cache_hit": False, "attempts": attempts, "request_hash": request_hash}
            start = time.time()
            usage, status, parsed, error, response_metadata = None, "failed", None, None, {}
            retryable = True
            try:
                response = self.client.chat.completions.create(
                    model=config.model, messages=messages, temperature=config.temperature,
                    max_tokens=config.max_tokens, timeout=config.timeout_seconds,
                    response_format={"type": "json_object"}, extra_body={"thinking": {"type": "disabled"}},
                )
                usage = extract_usage(response)
                response_metadata = {"id": response.id, "model": response.model,
                                     "system_fingerprint": getattr(response, "system_fingerprint", None)}
                choice = response.choices[0]
                response_metadata["finish_reason"] = choice.finish_reason
                if choice.finish_reason != "stop":
                    raise ValueError("Incomplete completion")
                parsed = json.loads(choice.message.content or "")
                validate(parsed)
                status = "ok"
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                error = {"type": type(exc).__name__, "http_status": code}
                # Do not serialize exception messages: SDK exceptions can include
                # request bodies or credential-bearing proxy details.
                retryable = code is None or code == 429 or code >= 500
                if code in (401, 403):
                    status = "authentication_failed"
            end = time.time()
            settled = self.ledger.settle(attempt_id, reserved, usage, config, start, end, status)
            attempts.append({**settled, "error": error, "response": response_metadata})
            result = {"status": status, "parsed": parsed if status == "ok" else None,
                      "attempts": attempts, "cache_hit": False, "request_hash": request_hash}
            if status == "ok":
                payload = {"request": request, "result": result}
                atomic_json(cache, {"payload": payload, "sha256": fingerprint(payload)})
                return result
            if not retryable:
                return result
            if index + 1 < config.max_attempts:
                time.sleep(0.5 * (index + 1))
        return result
