"""Runnable local demo (require.md Deliverables #1: "a runnable demo, local
or cloud"). A thin interactive wrapper around the existing pipeline -- no
new retrieval/generation logic here, just a multi-turn loop that keeps
`conversation_history` growing across questions within one run, which is
what actually demonstrates require.md's "conversation continuity within
the same session" requirement (a single question-answer-exit script
wouldn't).

Provider (local Ollama vs. OpenAI) is whatever `LLM_PROVIDER` is already
set to when this starts -- see `src/generation/providers.py`. This file
deliberately does nothing provider-specific: no API-key prompts, no writing
to the environment. Set `LLM_PROVIDER`/`OPENAI_API_KEY` yourself before
running if you want the OpenAI path; the local default needs no key at all.

Usage:
    uv run python -m src.demo
"""
from __future__ import annotations

import sys

from src.generation.llm import GenerationLLM
from src.generation.logging_utils import new_request_id
from src.generation.pipeline import answer_question
from src.generation.providers import BASE_URL, GENERATION_MODEL, PROVIDER, llm_client
from src.retrieval.retriever import get_default_retriever

EXIT_COMMANDS = {"exit", "quit", "q"}

REFUSAL_MESSAGES = {
    "low_retrieval_confidence": "(refused: nothing relevant enough was found in the knowledge base)",
    "llm_judged_insufficient_context": "(refused: the retrieved documents don't sufficiently answer this)",
}


def health_check() -> bool:
    print(f"Checking {PROVIDER} provider ({GENERATION_MODEL} @ {BASE_URL or 'api.openai.com'})...", file=sys.stderr)
    try:
        llm_client().models.list()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\nCouldn't reach the {PROVIDER} provider: {type(e).__name__}: {e}", file=sys.stderr)
        if PROVIDER == "local":
            print("Is `ollama serve` running, with mistral-16k pulled? See src/generation/providers.py.", file=sys.stderr)
        else:
            print("Check OPENAI_API_KEY is set and valid.", file=sys.stderr)
        return False


def format_answer(result) -> str:
    lines = [result.answer]
    if result.refused and result.refusal_reason in REFUSAL_MESSAGES:
        lines.append(REFUSAL_MESSAGES[result.refusal_reason])
    elif result.citations:
        lines.append(f"[sources: {', '.join(result.citations)}]")
    lines.append(f"({result.latency_ms['total'] / 1000:.1f}s)")
    return "\n".join(lines)


def main(argv=None) -> int:
    if not health_check():
        return 1

    print("Loading retriever (BGE-M3 + Chroma index)...", file=sys.stderr)
    retriever = get_default_retriever()
    llm = GenerationLLM()
    session_id = new_request_id()

    print(f"\nReady. Ask a question (English or Chinese), or type 'exit' to quit. [session: {session_id}]\n")

    conversation_history: list[dict] = []
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Exiting.")
            return 0

        print("(thinking -- local generation can take up to ~2 minutes)", file=sys.stderr)
        try:
            result = answer_question(
                conversation_history, question, retriever, llm, log=True, session_id=session_id
            )
        except Exception as e:  # noqa: BLE001
            print(f"Error: {type(e).__name__}: {e}\n", file=sys.stderr)
            continue

        print(format_answer(result) + "\n")
        conversation_history.append({"role": "user", "utterance": question})
        conversation_history.append({"role": "agent", "utterance": result.answer})


if __name__ == "__main__":
    raise SystemExit(main())
