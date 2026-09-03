"""LLM provider config -- local (Ollama) or OpenAI, one switch (LLM_PROVIDER
env var, defaults to "local").

Ollama was adopted as a stopgap when OpenAI credits ran out mid-project, not
as a permanent replacement -- the project expects to switch back once
credits are available, so both providers are kept as live, parallel configs
rather than one replacing the other in place. Every caller in this package
(llm.py, query_rewrite.py, run_eval.py) imports the resolved constants below
and never branches on provider itself.

To switch: `export LLM_PROVIDER=openai` (needs OPENAI_API_KEY set), or leave
it unset / `export LLM_PROVIDER=local` (needs `ollama serve` running).

--- Local provider notes ---

Two Ollama bugs (confirmed empirically, v0.32.11) shaped the local model
choices:
  1. Ollama's default runtime context window is 4096 tokens regardless of a
     model's architectural max, and our real prompts (system prompt + ~10
     retrieved chunks + history) commonly run 2-8K tokens -- so the default
     silently truncates input.
  2. The *fix* for #1 -- passing `options.num_ctx` per-request -- is itself
     silently ignored by Ollama's OpenAI-compatible endpoint (confirmed via
     a needle-in-haystack test: a fact placed at the start of an 8K-token
     prompt was lost with the override, found without it, and the endpoint
     still reported truncated prompt_tokens even though the override was
     sent). Only the native /api/chat endpoint honors runtime options.
  So the context window is baked into custom Modelfiles instead
  (`PARAMETER num_ctx 16384`, 2x our worst observed prompt size) -- that's
  respected no matter which endpoint loads the model. Hence the "-16k"
  model names; see e.g. `ollama show mistral-16k` for the resulting
  Modelfile.

  - generation (mistral-16k) -- qwen3.5:4b was tried first but has Ollama
    "thinking" capability that's always-on: it emits a hidden `reasoning`
    field before `content`, even under strict JSON-schema mode, and even
    `think: false` doesn't suppress it (same OpenAI-compat-endpoint bug as
    above -- only the native API honors that flag too). On short prompts
    this just adds latency, but on our real prompts reasoning can consume
    the whole completion budget before the model reaches `content`,
    producing an empty string that fails JSON parsing. This project already
    decided against CoT for generation (prompt.py), so a plain non-reasoning
    model is the better fit here, not a workaround. mistral:latest has no
    "thinking" capability (confirmed via /api/show) and was already pulled.
    Known trade-off, accepted and documented rather than chased further:
    generation latency measured at 48-120s/call on this hardware (fully
    GPU-resident -- not a CPU-fallback issue), far over require.md's <=10s
    p90 target. The OpenAI provider is the one expected to actually meet
    that target; this one trades latency for zero cost.
  - judge (qwen2.5-16k) is a different release line than the generator
    (avoids self-grading), doesn't have hidden-reasoning behavior (RAGAS's
    own prompts -- statement extraction, NLI checks -- are free-form and
    would have the same truncation risk), and is the exact model this
    dataset's own creators used to machine-translate its Chinese content
    (VALIDATION.md), so it's a known quantity on this corpus's language mix.
  - embedding (nomic-embed-text) is the only embedding model pulled
    locally; only used for RAGAS's answer_relevancy metric, not retrieval
    (retrieval stays on BGE-M3, unrelated to this evaluation-only need).
    Left off the -16k treatment -- embedding inputs here are single Q&A
    strings, not multi-chunk context blocks, so 4096 is not a real risk.
  - rewrite (llama3.2-16k) is the smallest available model -- query
    rewriting is a lightweight condensation task, not worth a bigger model,
    and it sits on the critical path before retrieval. Given the -16k
    treatment anyway since long multi-turn conversations could plausibly
    exceed 4096, and the cost of baking in headroom is free at request time.

--- OpenAI provider notes ---

Kept deliberately simple -- one model (gpt-5.4-mini, the project's earlier
choice) for all four roles, since this provider isn't the active one right
now. Retune per-role (e.g. a distinct judge model, to restore the
avoid-self-grading property the local provider has) if/when it's switched
back on. embedding uses text-embedding-3-small, OpenAI's standard low-cost
embedding model -- never actually exercised yet since the RAGAS integration
landed after the switch to local.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class ModelSet:
    generation: str
    judge: str
    embedding: str
    rewrite: str


_LOCAL = ModelSet(
    generation="mistral-16k",
    judge="qwen2.5-16k",
    embedding="nomic-embed-text",
    rewrite="llama3.2-16k",
)
_OPENAI = ModelSet(
    generation="gpt-5.4-mini",
    judge="gpt-5.4-mini",
    embedding="text-embedding-3-small",
    rewrite="gpt-5.4-mini",
)

PROVIDER = os.environ.get("LLM_PROVIDER", "local")
if PROVIDER not in ("local", "openai"):
    raise ValueError(f"LLM_PROVIDER must be 'local' or 'openai', got {PROVIDER!r}")

if PROVIDER == "local":
    _models = _LOCAL
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"  # unused by Ollama, but the OpenAI client requires a non-empty value
else:
    _models = _OPENAI
    BASE_URL = None  # None = the OpenAI SDK's own default (api.openai.com)
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if not API_KEY:
        raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")

GENERATION_MODEL = _models.generation
JUDGE_MODEL = _models.judge
EMBEDDING_MODEL = _models.embedding
REWRITE_MODEL = _models.rewrite


def llm_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)
