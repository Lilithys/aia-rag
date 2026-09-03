"""Query rewriting: condense multi-turn conversation history + a follow-up
question into one standalone retrieval query.

This is the A/B test queued from the retrieval-strategy discussion: does
rewriting beat the naive "join every turn + question" query construction
used everywhere else in this project? A cheap/fast model is deliberate --
this sits on the critical path before retrieval even starts, so it should
add minimal latency, unlike the main generation call.
"""
from __future__ import annotations

from .providers import REWRITE_MODEL, llm_client
from .retry import with_retry

_PROMPT_TEMPLATE = """Rewrite the follow-up question as a standalone question that includes all context needed to understand it on its own, using the conversation below. Keep it in the same language as the follow-up question. Output ONLY the rewritten question, nothing else -- no preamble, no quotes.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


class QueryRewriter:
    def __init__(self, model: str = REWRITE_MODEL):
        self.model = model
        self._client = llm_client()

    def rewrite(self, conversation_history: list[dict], question: str) -> str:
        if not conversation_history:
            return question  # nothing to condense
        history = "\n".join(f"{t['role']}: {t['utterance']}" for t in conversation_history)
        prompt = _PROMPT_TEMPLATE.format(history=history, question=question)
        resp = with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or question
