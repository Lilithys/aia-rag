"""Chat completion wrapper, structured-output mode. Provider (local Ollama
or OpenAI) is resolved by providers.py; this module doesn't care which."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .providers import GENERATION_MODEL, llm_client
from .prompt import CITATION_SCHEMA, SYSTEM_PROMPT, detect_language_hint, format_context
from .retry import with_retry


@dataclass
class LLMResult:
    answer: str
    citations: list[str]
    sufficient_context: bool
    prompt_tokens: int
    completion_tokens: int


class GenerationLLM:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model
        self._client = llm_client()

    def generate(
        self, conversation_history: list[dict], question: str, context_chunks, temperature: float | None = None
    ) -> LLMResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in conversation_history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["utterance"]})
        context_block = format_context(context_chunks)
        lang_hint = detect_language_hint(question)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n\n{context_block}\n\n---\n\n"
                    f"Question (respond in {lang_hint}): {question}"
                ),
            }
        )

        extra_kwargs = {} if temperature is None else {"temperature": temperature}
        resp = with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "grounded_answer", "schema": CITATION_SCHEMA, "strict": True},
                },
                **extra_kwargs,
            )
        )
        parsed = json.loads(resp.choices[0].message.content)
        return LLMResult(
            answer=parsed["answer"],
            citations=parsed["citations"],
            sufficient_context=parsed["sufficient_context"],
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
