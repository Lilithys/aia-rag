"""Re-score the first run's gpt-5.4-mini answers with RAGAS, using the same
local judge (qwen2.5-16k) as the mistral-16k run -- gives a same-judge,
apples-to-apples comparison without spending any OpenAI credits: no
generation happens here, only scoring of answers that already exist.

Source data, both pre-existing from before the local-model pivot:
  - `data/logs/generation.jsonl` -- append-only request log, still has all
    282 gpt-5.4-mini entries (redacted answer text, retrieved_chunk_ids)
    from before the pivot cutoff, untouched by later local-model runs.
  - `reports/04_generation_evaluation_openai_baseline.html` -- the first
    run's published report, with the exact 100 (prompt_tokens,
    completion_tokens, latency_ms) fingerprints for that specific run,
    embedded in its `const DATA = {...}` block.

The log has *multiple* gpt-5.4-mini passes mixed together (the original
100-question run, plus later RAGAS+OpenAI integration testing before
credits ran out) -- naive matching by reconstructed query text alone hits
multiple candidates for 97/100 questions. Matching on the baseline report's
per-question (prompt_tokens, completion_tokens, latency_ms.total) triple
instead resolves all 100 uniquely, since that triple is specific to one
exact generation call.

Usage:
    uv run python -m src.generation.rescore_openai_baseline
"""
from __future__ import annotations

import json
import re
import sys
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from src.generation.providers import API_KEY, BASE_URL, EMBEDDING_MODEL, JUDGE_MODEL

BASELINE_REPORT_PATH = "reports/04_generation_evaluation_openai_baseline.html"
LOG_PATH = "data/logs/generation.jsonl"
CHUNKS_PATH = "data/processed/chunks/chunks.jsonl"
TEST_PATH = "doc/test/test.json"
OUT_PATH = "data/processed/generation_eval_results_openai_rescored.json"
OPENAI_MODEL_NAME = "gpt-5.4-mini"
MAX_WORKERS = 3
RAGAS_TIMEOUT = 900


def load_baseline_items() -> list[dict]:
    html = open(BASELINE_REPORT_PATH, encoding="utf-8").read()
    m = re.search(r"const DATA = (\{.*?\});", html)
    if not m:
        raise RuntimeError(f"could not find embedded DATA in {BASELINE_REPORT_PATH}")
    return json.loads(m.group(1))["items"]


def load_openai_logs() -> list[dict]:
    logs = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("model") == OPENAI_MODEL_NAME:
                logs.append(d)
    return logs


def match_baseline_to_logs(baseline: list[dict], logs: list[dict]) -> dict[str, dict | None]:
    """Fingerprint-match each baseline question to its exact log entry via
    (prompt_tokens, completion_tokens, latency_ms.total) -- see module
    docstring for why naive query-text matching isn't precise enough here.
    """

    def fp(prompt_tok, completion_tok, latency_total):
        return (prompt_tok, completion_tok, round(latency_total, 1))

    by_fp: dict[tuple, list[dict]] = {}
    for log in logs:
        key = fp(log["token_usage"].get("prompt_tokens", -1), log["token_usage"].get("completion_tokens", -1), log["latency_ms"]["total"])
        by_fp.setdefault(key, []).append(log)

    matches: dict[str, dict | None] = {}
    for item in baseline:
        if item["refused"] and item["prompt_tokens"] == 0:
            matches[item["question_id"]] = None  # low-confidence refusal never called the LLM
            continue
        key = fp(item["prompt_tokens"], item["completion_tokens"], item["latency_ms"]["total"])
        hits = by_fp.get(key, [])
        matches[item["question_id"]] = hits[0] if hits else None
        if len(hits) > 1:
            print(f"  warning: {len(hits)} candidate logs for {item['question_id']}, took the first", file=sys.stderr)
    return matches


def build_ragas_rows(baseline: list[dict], matches: dict[str, dict | None]) -> list[dict]:
    test_by_qid = {it["question_id"]: it for it in json.load(open(TEST_PATH, encoding="utf-8"))["items"]}
    chunk_text = {json.loads(l)["chunk_id"]: json.loads(l)["text"] for l in open(CHUNKS_PATH, encoding="utf-8")}

    rows = []
    for item in baseline:
        qid = item["question_id"]
        log = matches.get(qid)
        test_item = test_by_qid[qid]
        if log is None or log["refused"] or not test_item.get("answerable", True) or not test_item.get("gold_answer"):
            continue
        contexts = [chunk_text[cid] for cid in log["retrieved_chunk_ids"] if cid in chunk_text]
        rows.append(
            {
                "question_id": qid,
                "user_input": test_item["question"],
                "response": log["answer_redacted"],
                "retrieved_contexts": contexts,
                "reference": test_item["gold_answer"],
            }
        )
    return rows


def run_ragas_rescore(rows: list[dict]) -> dict[str, dict]:
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, base_url=BASE_URL, api_key=API_KEY))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL, api_key=API_KEY, check_embedding_ctx_length=False)
    )
    dataset = EvaluationDataset.from_list(
        [{"user_input": r["user_input"], "response": r["response"], "retrieved_contexts": r["retrieved_contexts"], "reference": r["reference"]} for r in rows]
    )
    result = evaluate(
        dataset,
        metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy(), AnswerAccuracy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=RAGAS_TIMEOUT),
        show_progress=False,
    )
    df = result.to_pandas()
    return {
        r["question_id"]: {
            "context_precision": row["context_precision"],
            "context_recall": row["context_recall"],
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "answer_accuracy": row["nv_accuracy"],
        }
        for r, row in zip(rows, df.to_dict("records"))
    }


def main(argv=None) -> int:
    baseline = load_baseline_items()
    logs = load_openai_logs()
    print(f"Loaded {len(baseline)} baseline questions, {len(logs)} {OPENAI_MODEL_NAME} log entries.", file=sys.stderr)

    matches = match_baseline_to_logs(baseline, logs)
    rows = build_ragas_rows(baseline, matches)
    print(f"Built {len(rows)} eligible RAGAS rows.", file=sys.stderr)

    t0 = time.time()
    scores = run_ragas_rescore(rows)
    print(f"RAGAS wall time: {time.time() - t0:.1f}s", file=sys.stderr)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"config": {"model": OPENAI_MODEL_NAME, "ragas_judge_model": JUDGE_MODEL, "ragas_embedding_model": EMBEDDING_MODEL, "n_rows": len(rows)}, "scores": scores},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(scores)} scores to {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
