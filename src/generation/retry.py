"""Retry wrapper for OpenAI-client calls under concurrent load.

Originally written for the hosted OpenAI API's RateLimitError (the SDK's own
2-retry default wasn't enough at high concurrency against a 200K TPM limit).
Now serving local Ollama calls too, where there's no quota to exhaust but a
single local server can still return connection/timeout errors under
concurrent load (several requests queued against one GPU/CPU) -- so this
retries both error families rather than assuming one deployment target.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, RateLimitError

T = TypeVar("T")
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


def with_retry(fn: Callable[[], T], max_retries: int = 6, base_delay: float = 2.0) -> T:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except _RETRYABLE as e:
            last_exc = e
            delay = base_delay * (2**attempt)
            time.sleep(min(delay, 30.0))
    raise last_exc
