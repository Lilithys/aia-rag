"""Temperature sensitivity analysis (require.md Global Constraints #3: cost
sensitivity analysis for top_k, reranker on/off, and temperature -- >=3
settings; top_k and reranker were covered in step 4, this covers temperature).

Verified first, not assumed: a quick check (3 repeated calls each at
temperature=0.0 and 1.3 on an identical prompt) confirmed the parameter
actually takes effect through Ollama's OpenAI-compatible endpoint --
temp=0.0 gave byte-identical output every time, temp=1.3 gave varied
phrasing -- unlike `options.num_ctx` and `think`, which that same endpoint
silently ignores (see providers.py). Worth checking every time a new
parameter crosses that endpoint; two of three tried so far were broken.

Runs a stratified 15-item subset (not the full 101 -- local generation
alone is 48-120s/call, so 3 settings x full set would take many hours for
a sensitivity check) across three temperatures: 0.0 (deterministic),
0.7 (a common default), 1.3 (high). Reuses run_eval.py's run_pipeline/
run_ragas so results are directly comparable to the main evaluation.

Usage:
    uv run python -m src.generation.eval_temperature
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

from src.generation.llm import GenerationLLM
from src.generation.pipeline import answer_question
from src.generation.run_eval import run_ragas
from src.retrieval.retriever import get_default_retriever

SUBSET_IDS_PATH = "/private/tmp/claude-501/-Users-yishanma-Desktop-aia-rag/b07484ec-d49d-4d39-8428-1b849a2d8bd5/scratchpad/temp_sensitivity_subset_ids.json"
OUT_PATH = "data/processed/temperature_sensitivity_results.json"
TEMPERATURES = [0.0, 0.7, 1.3]
MAX_WORKERS = 3


def load_subset() -> list[dict]:
    items = json.load(open("doc/test/test.json", encoding="utf-8"))["items"]
    ids = json.load(open(SUBSET_IDS_PATH))
    by_qid = {it["question_id"]: it for it in items}
    return [by_qid[qid] for qid in ids]


def run_one_temperature(items: list[dict], retriever, llm: GenerationLLM, temperature: float) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n--- temperature={temperature} ---", file=sys.stderr)
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                answer_question,
                item["conversation_history"],
                item["question"],
                retriever,
                llm,
                log=True,
                session_id=f"temp-sensitivity-{temperature}-{item['question_id']}",
                temperature=temperature,
            ): item
            for item in items
        }
        for i, fut in enumerate(as_completed(futures), 1):
            item = futures[fut]
            try:
                results[item["question_id"]] = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {item['question_id']}: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"  [{i}/{len(items)}] generated", file=sys.stderr)

    ragas_scores = run_ragas(items, results)

    citations_valid = [r.citations_valid for r in results.values() if not r.refused]
    latencies = [r.latency_ms["total"] for r in results.values()]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "temperature": temperature,
        "n_items": len(items),
        "n_refused": sum(1 for r in results.values() if r.refused),
        "citation_validity_rate": mean(citations_valid),
        "latency_p50_s": round(sorted(latencies)[len(latencies) // 2] / 1000, 1) if latencies else None,
        "context_precision": mean([s["context_precision"] for s in ragas_scores.values()]),
        "context_recall": mean([s["context_recall"] for s in ragas_scores.values()]),
        "faithfulness": mean([s["faithfulness"] for s in ragas_scores.values()]),
        "answer_relevancy": mean([s["answer_relevancy"] for s in ragas_scores.values()]),
        "answer_accuracy": mean([s["answer_accuracy"] for s in ragas_scores.values()]),
        "n_ragas_scored": len(ragas_scores),
        "per_item": {
            qid: {**asdict(r), "ragas": ragas_scores.get(qid)}
            for qid, r in results.items()
        },
    }


def main(argv=None) -> int:
    items = load_subset()
    print(f"Loaded {len(items)} stratified subset items.", file=sys.stderr)

    retriever = get_default_retriever()
    llm = GenerationLLM()

    all_results = []
    for temp in TEMPERATURES:
        t0 = time.time()
        result = run_one_temperature(items, retriever, llm, temp)
        result["wall_seconds"] = round(time.time() - t0, 1)
        all_results.append(result)
        print(
            f"temperature={temp}: faithfulness={result['faithfulness']}, "
            f"accuracy={result['answer_accuracy']}, citation_validity={result['citation_validity_rate']}, "
            f"latency_p50={result['latency_p50_s']}s, wall={result['wall_seconds']}s",
            file=sys.stderr,
        )

    import os

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump({"subset_size": len(items), "results": all_results}, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nWrote results to {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
