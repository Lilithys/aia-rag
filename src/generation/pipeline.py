"""End-to-end pipeline: retrieve -> confidence gate -> generate -> guardrail -> log.

Confidence threshold (0.55) is a cheap pre-filter for obviously-off-topic
queries only -- per the analysis behind this step, answerable/unanswerable
top-1 similarity distributions overlap substantially on this corpus (this
benchmark deliberately includes high-similarity hard negatives), so a single
retrieval-score threshold can't cleanly separate the two. The real
answerable/unanswerable judgment happens inside the LLM call
(`sufficient_context`), which sees the actual retrieved text, not just a
similarity number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.chunking.strategies.base import ChunkRecord

from .llm import GenerationLLM
from .logging_utils import LogEntry, log_interaction, new_request_id, redact_entry_text

CONFIDENCE_THRESHOLD = 0.55
TOP_K = 10


@dataclass
class AnswerResult:
    answer: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    chunks_used: list[ChunkRecord]
    request_id: str
    latency_ms: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Whether every citation the LLM originally emitted matched a retrieved
    # chunk_id, checked *before* the invalid ones below are filtered out of
    # `citations`. A caller checking "all(cid in valid_ids for cid in
    # citations)" against the already-filtered list would get True by
    # construction regardless of what the model actually emitted -- this
    # field is what actually answers "did the model hallucinate a citation".
    citations_valid: bool = True


def _build_retrieval_query(conversation_history: list[dict], question: str) -> str:
    turns = [t["utterance"] for t in conversation_history]
    return "\n".join(turns + [question])


def answer_question(
    conversation_history: list[dict],
    question: str,
    retriever,
    llm: GenerationLLM,
    top_k: int = TOP_K,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    log: bool = True,
    session_id: str | None = None,
    temperature: float | None = None,
) -> AnswerResult:
    request_id = new_request_id()
    session_id = session_id or request_id
    t_start = time.time()

    query = _build_retrieval_query(conversation_history, question)
    t0 = time.time()
    scored = retriever.search_with_scores(query, top_k=top_k)
    retrieval_ms = (time.time() - t0) * 1000
    chunks = [c for c, _ in scored]
    top1_score = scored[0][1] if scored else 0.0

    if top1_score < confidence_threshold:
        result = AnswerResult(
            answer="I don't have enough relevant information in my knowledge base to answer this question confidently. Could you rephrase, or ask about a different topic?",
            citations=[],
            refused=True,
            refusal_reason="low_retrieval_confidence",
            chunks_used=[],
            request_id=request_id,
            latency_ms={"retrieval": round(retrieval_ms, 1), "generation": 0.0, "total": round((time.time() - t_start) * 1000, 1)},
        )
        if log:
            _log(request_id, session_id, query, top1_score, [], result, usage={}, citations_valid=True, model=None)
        return result

    t0 = time.time()
    llm_result = llm.generate(conversation_history, question, chunks, temperature=temperature)
    generation_ms = (time.time() - t0) * 1000

    valid_ids = {c.chunk_id for c in chunks}
    citations_valid_list = [cid for cid in llm_result.citations if cid in valid_ids]
    all_citations_valid = citations_valid_list == llm_result.citations

    latency_ms = {
        "retrieval": round(retrieval_ms, 1),
        "generation": round(generation_ms, 1),
        "total": round((time.time() - t_start) * 1000, 1),
    }
    if not llm_result.sufficient_context:
        result = AnswerResult(
            answer=llm_result.answer,
            citations=[],
            refused=True,
            refusal_reason="llm_judged_insufficient_context",
            chunks_used=chunks,
            request_id=request_id,
            latency_ms=latency_ms,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
        )
    else:
        result = AnswerResult(
            answer=llm_result.answer,
            citations=citations_valid_list,
            refused=False,
            refusal_reason=None,
            chunks_used=chunks,
            request_id=request_id,
            latency_ms=latency_ms,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            citations_valid=all_citations_valid,
        )

    if log:
        _log(
            request_id,
            session_id,
            query,
            top1_score,
            chunks,
            result,
            usage={"prompt_tokens": llm_result.prompt_tokens, "completion_tokens": llm_result.completion_tokens},
            citations_valid=all_citations_valid,
            model=llm.model,
        )
    return result


def _log(request_id, session_id, query, top1_score, chunks, result: AnswerResult, usage, citations_valid, model):
    query_r, answer_r = redact_entry_text(query, result.answer)
    entry = LogEntry(
        timestamp=time.time(),
        request_id=request_id,
        session_id=session_id,
        query_redacted=query_r,
        retrieval_top1_score=round(top1_score, 4),
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        retrieved_doc_ids=sorted({c.doc_id for c in chunks}),
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        answer_redacted=answer_r,
        citations=result.citations,
        citations_valid=citations_valid,
        latency_ms=result.latency_ms,
        token_usage=usage,
        model=model,
    )
    log_interaction(entry)
