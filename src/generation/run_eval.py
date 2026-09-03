"""Step 5 evaluation: run the full retrieve->generate pipeline over all 100
test questions and measure RAG quality with RAGAS (context_precision,
context_recall, faithfulness, answer_relevancy, answer_accuracy -- the
standard definitions, not a hand-built judge) plus latency and citation
validity, which RAGAS doesn't define and stay custom.

Runs against whichever provider `src/generation/providers.py` is currently
configured for (local Ollama by default, no API key/cost -- see that module
for why each model was assigned the role it has, and for the local-model
latency trade-off this evaluation is currently accepting). RAGAS's own
`llm_factory`/`embedding_factory` helpers don't expose a `base_url`, so the
LLM/embeddings are constructed directly via langchain_openai instead, using
the same BASE_URL/API_KEY the rest of this package resolves from that
provider config.

Two phases: (1) run the pipeline concurrently (ThreadPoolExecutor); (2)
batch-score every non-refused, answerable, gold-having answer with RAGAS's
own `evaluate()`, which does its own internal concurrency. Concurrency is
kept modest in both phases: a local Ollama server has one shared GPU/CPU
behind it (confirmed serialized to one in-flight request at a time, via
`-np 1` in the underlying llama-server process), so parallel requests
contend for the same compute rather than getting elastic capacity the way
a hosted API would. RAGAS's per-call timeout is also raised well above its
180s default for the same reason -- see RAGAS_TIMEOUT below.

Usage:
    uv run python -m src.generation.run_eval
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from src.generation.llm import GenerationLLM
from src.generation.pipeline import CONFIDENCE_THRESHOLD, TOP_K, AnswerResult, answer_question
from src.generation.providers import API_KEY, BASE_URL, EMBEDDING_MODEL, JUDGE_MODEL
from src.retrieval.retriever import get_default_retriever

OUT_PATH = "data/processed/generation_eval_results.json"
MAX_WORKERS = 3
# RAGAS's default (180s) is tuned for a hosted API; against a local model
# under -np 1 (one request actually processed at a time regardless of
# max_workers -- confirmed via `ollama ps`/process args), a call that's
# queued behind others can blow past 180s before it even starts, not just
# while running. Confirmed by a smoke test: 3/4 items came back with
# TimeoutError-driven NaNs on context_precision (the highest-call-volume
# metric, one verification call per retrieved chunk) at the 180s default.
RAGAS_TIMEOUT = 900


@dataclass
class ItemResult:
    question_id: str
    answerable: bool
    language_relation: str | None
    refused: bool
    refusal_reason: str | None
    citations: list[str]
    citations_valid: bool
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    answer_accuracy: float | None
    latency_ms: dict
    prompt_tokens: int
    completion_tokens: int


def _run_pipeline_one(item: dict, retriever, llm: GenerationLLM) -> AnswerResult:
    # Each test item bundles its whole conversation into one call rather than
    # simulating real back-and-forth turns, so question_id doubles as a
    # stable session_id -- it lets a log line be traced straight back to the
    # test item that produced it, which a random request_id alone can't do.
    return answer_question(
        item["conversation_history"], item["question"], retriever, llm, log=True, session_id=item["question_id"]
    )


def run_pipeline(items: list[dict], retriever, llm: GenerationLLM) -> dict[str, AnswerResult]:
    results: dict[str, AnswerResult] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_pipeline_one, item, retriever, llm): item for item in items}
        for i, fut in enumerate(as_completed(futures), 1):
            item = futures[fut]
            try:
                results[item["question_id"]] = fut.result()
            except Exception as e:  # noqa: BLE001 - keep going, report failures
                print(f"  FAILED {item.get('question_id')}: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"[{i}/{len(items)}] generated", file=sys.stderr)
    return results


def run_ragas(items: list[dict], pipeline_results: dict[str, AnswerResult]) -> dict[str, dict]:
    """Scores every eligible item with RAGAS, returns {question_id: {metric: score}}."""
    rows = []
    row_ids = []
    for item in items:
        r = pipeline_results.get(item["question_id"])
        if r is None or r.refused or not item.get("answerable", True) or not item.get("gold_answer"):
            continue
        rows.append(
            {
                "user_input": item["question"],
                "response": r.answer,
                "retrieved_contexts": [c.text for c in r.chunks_used],
                "reference": item["gold_answer"],
            }
        )
        row_ids.append(item["question_id"])

    if not rows:
        return {}

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, base_url=BASE_URL, api_key=API_KEY))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL, api_key=API_KEY, check_embedding_ctx_length=False)
    )
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy(), AnswerAccuracy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=RAGAS_TIMEOUT),
        show_progress=False,
    )
    df = result.to_pandas()

    out = {}
    for qid, row in zip(row_ids, df.to_dict("records")):
        out[qid] = {
            "context_precision": row["context_precision"],
            "context_recall": row["context_recall"],
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "answer_accuracy": row["nv_accuracy"],  # AnswerAccuracy's internal metric name
        }
    return out


def main(argv=None) -> int:
    items = json.load(open("doc/test/test.json", encoding="utf-8"))["items"]
    print(f"Loaded {len(items)} test questions.", file=sys.stderr)

    retriever = get_default_retriever()
    llm = GenerationLLM()

    t0 = time.time()
    pipeline_results = run_pipeline(items, retriever, llm)
    print(f"Pipeline wall time: {time.time() - t0:.1f}s", file=sys.stderr)

    print("Scoring with RAGAS...", file=sys.stderr)
    t0 = time.time()
    ragas_scores = run_ragas(items, pipeline_results)
    print(f"RAGAS wall time: {time.time() - t0:.1f}s ({len(ragas_scores)} items scored)", file=sys.stderr)

    item_results: list[ItemResult] = []
    for item in items:
        qid = item["question_id"]
        r = pipeline_results.get(qid)
        if r is None:
            continue
        scores = ragas_scores.get(qid, {})
        item_results.append(
            ItemResult(
                question_id=qid,
                answerable=item.get("answerable", True),
                language_relation=item.get("language_relation"),
                refused=r.refused,
                refusal_reason=r.refusal_reason,
                citations=r.citations,
                citations_valid=r.citations_valid,
                context_precision=scores.get("context_precision"),
                context_recall=scores.get("context_recall"),
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
                answer_accuracy=scores.get("answer_accuracy"),
                latency_ms=r.latency_ms,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
            )
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "top_k": TOP_K,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "model": llm.model,
                    "ragas_judge_model": JUDGE_MODEL,
                    "ragas_embedding_model": EMBEDDING_MODEL,
                },
                "items": [asdict(r) for r in item_results],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(item_results)} results to {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
