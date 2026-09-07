# Project Summary — RAG QA Case Study

> 历史实验记录：保留当时的结论、参数和失败。GitHub 提交版已删除旧 runner、盲评表和重复缓存；本文件内历史命令及产物清单不代表当前运行接口。当前入口与复现范围见仓库 README。

Status as of 2026-09-05. This document is a chronological/thematic account of every key decision, experiment, and result produced so far, with the exact numbers behind each. It complements (and in a couple of places corrects/updates) [`DESIGN_NOTE.md`](DESIGN_NOTE.md), which is the required 200–500 word rationale for the final deliverable — see [Open items](#16-open-items--known-doc-gaps) for where the two have drifted apart.

## 1. Brief

Per [`require.md`](../require.md): a retrieval-augmented QA system over a bilingual (CN/EN) internal knowledge base, supporting multi-turn dialogue and grounded, citation-backed answers. Hard quantitative gates: **Faithfulness ≥ 0.85**, **Context Precision ≥ 0.70**, **overall answer accuracy ≥ 80%** on a self-built eval set, **p90 end-to-end latency ≤ 10s**, plus a token-cost estimate with top_k/reranker/temperature sensitivity analysis, a logging system, minimal PII/prompt-injection handling, and a 200–500 word design note.

## 2. Corpus & Test Set

The working corpus is a self-built subset of **MultiDoc2Dial**: 91 documents / 101 test questions (97 answerable, 4 unanswerable — see [§11](#11-key-bugs-found--fixed) for how this set was corrected from its original 100/15 split).

- **Formats:** 68 Markdown, 9 text-layer PDF, 5 image-only scanned PDF, 5 DOCX, 4 UTF-8 TXT.
- **Languages:** 46 English, 27 Simplified Chinese, 18 Chinese-English mixed documents; questions split 50 EN / 30 CN / 20 mixed, with 22 answerable cross-lingual cases (question language ≠ required document language) deliberately included to stress-test embedding choice.
- **Retrieval structure:** 75 single-document + 10 two-document composite answerable questions, plus unanswerable cases; every question ships 2 hard / 2 medium / 2 easy negative document IDs (hard = same-domain, high lexical TF-IDF similarity; medium = same-domain, lower-ranked; easy = cross-domain) so retrieval quality is measured against real distractors, not just against the right answer being present somewhere.

A much larger second corpus (488 docs / 661 dialogues / 4335 eval pairs) was prepared later for a scale-up — see [§14](#14-full-corpus-scale-up-prepared-paused).

## 3. Pipeline Architecture

Five stages, each decided by measuring alternatives against this corpus rather than by default (`reports/01`–`04`):

```
raw docs → chunking → embedding → retrieval (+optional rerank) → generation (LLM + citations) → RAGAS evaluation
```

At this point in the project, generation ran behind one `LLM_PROVIDER` switch (`local` / `openai` / `deepseek`, see [§13](#13-multi-provider-architecture--deepseek-migration)); every other module imported resolved constants from `src/generation/providers.py`. **This has since been superseded** — the later rewrite (§17 onward) consolidated onto DeepSeek as the sole provider; `providers.py` and the `LLM_PROVIDER` switch no longer exist. Current provider config lives in `src/evaluation/paid_calls_v1.py` and `src/service/runtime.py` — see the root [`README.md`](../README.md) for the current setup.

## 4. Stage 1 — Chunking

27 configurations swept: 3 strategies (fixed-size, recursive-structural, semantic/section-aware) × 3 target sizes (256/512/1024 tokens) × 3 overlap ratios (0/15/30%), scored against 139 gold evidence spans and a BM25 retrieval proxy over the 85 answerable questions.

**Chosen: semantic/section-aware chunking, 1024-token target, no overlap → 199 chunks.** Per `reports/01_chunking_comparison.html`'s own recommendation: it posts the best MRR@10 of all 27 configurations (0.5975), *ties* for the best retrieval recall (semantic/fixed/recursive are all within ~3pp of each other at matched chunk size — semantic doesn't strictly win on recall alone), while using ~35% fewer tokens per chunk than fixed-size at the same nominal setting and keeping chunk boundaries aligned to actual document sections — which is what later made each chunk a clean, self-contained unit for citation and (as it turned out) for the reranker's context window ([§6](#6-stage-3--retrieval--reranking)). Overlap had negligible effect at every strategy/size (MRR/Recall@5 spread <2pp across 0/15/30%); semantic chunking was structurally the least sensitive to it, since overlap only applies to its rare oversized-section fallback.

| Strategy (1024 tok, no overlap) | # chunks | recall@5 | recall@10 | MRR@10 | evidence containment |
|---|---|---|---|---|---|
| Fixed-size | 144 | 71.8% | 74.1% | 0.573 | 100% |
| Recursive-structural | 153 | 69.4% | 71.8% | 0.579 | 100% |
| **Semantic/section (chosen)** | **199** | 69.4% | 69.4% | **0.598** | 99.3% |

## 5. Stage 2 — Embedding

Three candidates compared on the winning chunking config: BM25 (sparse baseline), **BGE-M3**, multilingual-e5-large.

| Model | recall@5 | recall@10 | MRR@10 | cross-lingual recall@5 |
|---|---|---|---|---|
| BM25 baseline | 69.4% | 69.4% | 0.598 | 0.0% |
| **BGE-M3 (chosen)** | **85.9%** | **96.5%** | **0.704** | **66.7%** |
| multilingual-e5-large | 67.1% | 75.3% | 0.595 | 5.6% |

Decided almost entirely by the cross-lingual gap (66.7% vs 5.6% recall@5) — this corpus is deliberately multilingual, and same-language quality wasn't the deciding factor; on aligned-language questions all three models are much closer. BGE-M3 embeds locally at ~3400 tokens/sec corpus-side, no API cost.

## 6. Stage 3 — Retrieval & Reranking

| Retrieval strategy | recall@5 | recall@10 | context_precision@10 |
|---|---|---|---|
| **Dense only (chosen)** | **85.9%** | **96.5%** | 12.2% |
| Sparse (BM25) | 69.4% | 69.4% | 8.2% |
| Hybrid (RRF fusion) | 69.4% | 76.5% | 9.8% |

Dense-only beat both sparse and hybrid outright. Reranking was tried on top of dense retrieval and made things *worse*, not better:

| Config | recall@5 | latency |
|---|---|---|
| No reranker (chosen) | 85.9% | 65.6ms |
| bge-reranker-**base** | 70.6% | 1109.5ms |
| bge-reranker-**large** (follow-up check) | 85.9% (recovers to no-reranker level) | ~3800ms (est.) |

The base reranker's regression was confirmed not to be "just a weak model" — the large variant recovers the same recall bge-reranker-base lost, but at ~58x the latency of no reranker for zero net quality gain. Root cause: the reranker's 512-token cross-encoder window truncates the same large (up to 1024-token), semantically coherent chunks that made chunking work well in the first place.

**top_k = 10** was chosen from a sweep (recall@1 57.7% → @3 82.4% → @5 85.9% → @10 96.5% → @20 100%) — 10 sits just below the recall ceiling without paying for @20's marginal 3.5pp at double the context.

## 7. Multi-turn Query Handling

A query-rewrite experiment (using conversation history to rewrite the follow-up question before retrieval, vs. the production baseline of simply flattening all turns + the question into one retrieval query) was run as part of the multi-turn requirement:

| | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| Baseline (flatten history, chosen) | 85.9% | **94.1%** | 0.704 |
| LLM-rewritten query | **91.8%** | 92.9% | **0.758** |

Rewriting helps recall@5 and MRR but very slightly *hurts* recall@10 (95.3%→92.9%... i.e. -1.2pp) — since production actually retrieves at top_k=10, not top_k=5, the baseline was kept: no measurable win at the k that matters, for the cost of an extra LLM call (~0.14s/query) per turn. Documented as a measured-but-not-adopted alternative, not an oversight.

## 8. Generation Pipeline Design

One structured-output call returns `answer`, `citations`, and `sufficient_context` together — no chain-of-thought, keeping latency down and making faithfulness a structural property (every claim must cite a retrieved `chunk_id`) rather than something hoped for. A low similarity threshold (0.55) is a cheap pre-filter for obviously off-topic queries only; answerable/unanswerable score distributions overlap too much on this corpus for a single threshold to safely do more, so the real answerable/unanswerable judgment happens inside the LLM call, which sees the actual retrieved text and can refuse via `sufficient_context: false`.

## 9. Generation Evaluation (RAGAS)

Standard RAGAS metrics replaced an earlier hand-built judge: **context_precision** (avg. precision over retrieved chunks), **context_recall**, **faithfulness**, **answer_relevancy**, **answer_accuracy** (`AnswerAccuracy`/`nv_accuracy`, a 2-call 0/2/4-scale average). Three generation-model configs have now been run end-to-end on the 101-item subset:

| Metric | mistral-16k (local) | gpt-5.4-mini (same-judge rescore) | **deepseek-v4-flash** | require.md target |
|---|---|---|---|---|
| context_precision | 56.7% | 57.0% | **70.7%*** | ≥ 70% |
| context_recall | 87.7% | 89.0% | 90.8% | — |
| faithfulness | 72.0% | 91.7% | 88.9%* | ≥ 85% |
| answer_relevancy | 42.3% | 59.9% | 73.3% | — |
| answer_accuracy | 25.0% | 38.2% | **80.5%** | ≥ 80% |
| citation validity | 51.6% (49/95) | — | 100% (86/86) | — |

\*Caveat: deepseek's row was RAGAS-judged by **deepseek-v4-pro**, while mistral-16k and gpt-5.4-mini were both judged by the same local judge (qwen2.5-16k) — which is exactly why *their* context_precision matches so closely (56.7% vs 57.0%) despite different generation models. Retrieval itself is identical across all three rows (same Chroma collection, same retriever, same top_k=10), so part of deepseek's context_precision jump is plausibly judge leniency, not better retrieval. A same-judge rescore (mirroring what was already done for gpt-5.4-mini) is the natural next step before treating this as a clean win — see [§16](#16-open-items--known-doc-gaps).

Gpt-5.4-mini's numbers come from **rescoring the original OpenAI-era answers** (still preserved in `data/logs/generation.jsonl`) with the same local RAGAS judge used for mistral-16k, at zero additional API cost — matched by a `(prompt_tokens, completion_tokens, latency.total)` fingerprint, 76/100 baseline questions matched uniquely.

**Latency vs. the require.md p90 ≤ 10s gate** (measured directly from logged per-item latencies, not estimated):

| | p50 | p90 | max | % of items ≤ 10s |
|---|---|---|---|---|
| mistral-16k (local) | 141.6s | 183.7s | 225.3s | 0% |
| deepseek-v4-flash | **9.1s** | 22.1s | 166.9s | 60% |

Local inference fails this gate outright (confirmed serialized to one in-flight request at a time on the Ollama server, `-np 1`). DeepSeek gets the median under budget but still misses the 90th-percentile target — a long tail (one item hit 166.9s) currently keeps it at 60%, not 90%, under 10s. Not yet root-caused to a specific mechanism; multi-turn conversations with many history turns are the leading suspect given earlier "lost in the middle" observations on this corpus, but this needs its own targeted look before claiming a cause.

## 10. Temperature Sensitivity (require.md-required, ≥3 settings)

15-item stratified subset, mistral-16k, temperatures 0.0 / 0.7 / 1.3:

| Temp | context_precision | context_recall | faithfulness | answer_relevancy | answer_accuracy | citation validity |
|---|---|---|---|---|---|---|
| 0.0 | 0.544 | 0.846 | 0.736 | 0.467 | **0.289** | 0.538 |
| 0.7 | 0.544 | 0.846 | **0.809** | **0.549** | 0.212 | 0.538 |
| 1.3 | 0.544 | 0.846 | 0.696 | 0.401 | 0.250 | **0.308** |

Context precision/recall are exactly identical across all three — confirms they're generation-temperature-independent (they're retrieval-stage metrics; this is a sanity check, not a surprise). Faithfulness and answer_relevancy both peak at 0.7 (non-monotonic — 0.0 is *not* the safest choice for these two). Answer_accuracy does **not** follow the same pattern — it's actually highest at 0.0 and dips at 0.7, so "0.7 is best" is not a blanket conclusion. Citation validity holds steady at 0.0/0.7 but drops sharply at 1.3 (0.538→0.308) — high temperature measurably increases citation hallucination.

## 11. Key Bugs Found & Fixed

- **Citation-hallucination tautology.** `citations_valid` was computed against `AnswerResult.citations`, which was *already filtered* to only valid IDs by the time the check ran — so it was always `True` by construction, regardless of what the model actually emitted. Fixed by adding a genuine pre-filter `citations_valid` field, computed against the model's raw, unfiltered citation list. This is what surfaced the real ~52% citation-validity rate on mistral-16k reported above — previously invisible.
- **Ollama silently ignoring `options.num_ctx` and `think:false`** on its OpenAI-compatible endpoint (only the native `/api/chat` endpoint honors them). Fixed with custom Modelfiles baking `PARAMETER num_ctx 16384` directly into `mistral-16k`/`qwen2.5-16k`/`llama3.2-16k`.
- **`qwen3.5:4b` reasoning model consuming the entire completion budget on its hidden `reasoning` field** before ever reaching `content`, producing empty answers. Switched generation to the non-reasoning `mistral-16k`.
- **`materialize.py` had no `--in-dir` CLI argument** — silently reprocessed the old 91-doc corpus even when a new corpus was intended. Caught because the resulting chunk/doc counts looked wrong; fixed by adding the flag.
- **Stratified sampling `ValueError: Sample larger than population`** in the new-corpus test-set converter — a proportional-allocation bug forced `max(1, …)` per stratum bucket regardless of remaining sample budget. Rewritten using largest-remainder allocation, verified correct at n=10/300/5000 (5000 correctly caps at the full population).
- **Test-set quality review.** Of the original 15 unanswerable-labeled questions, 12 were found to actually be answerable from the retrieved corpus (relabeled with drafted gold answers, evidence, and a correction note) and 3 were dropped as genuinely ambiguous. 4 new out-of-scope questions were added to keep a clean unanswerable-refusal test — the first draft (passport renewal, California DMV fees, etc.) scored *above* the 0.55 refusal threshold (0.55–0.69, i.e., too close to real corpus content to reliably test refusal), so they were replaced with genuinely unrelated topics (home repair, restaurants, fitness, weather), which scored 0.32–0.46. Net: 100→101 items, 85→97 answerable.

## 12. Security & Robustness

- **PII redaction:** regex-based, applied at the logging boundary only (the LLM itself still sees real input, since it needs to answer) — 8/8 test cases pass.
- **Prompt-injection red-team:** 3 adversarial cases embedded inside synthetic retrieved chunks, tested against mistral-16k. **2 of 3 succeeded**: a persona-hijack (got the model to comply in pirate-speak) and a fake-authority privilege-escalation (got the model to agree to share a stranger's SSN). A direct system-prompt-exfiltration attempt was resisted. Per require.md's "minimal" prompt-injection defense bar, this gap is **documented as known, not fixed** — explicit product decision, not an oversight — and listed as future work in `DESIGN_NOTE.md`.
- **Refusal design:** low-retrieval-confidence and LLM-judged-insufficient-context are both surfaced as explicit refusal reasons in logs and in `AnswerResult`, satisfying require.md's "explanatory refusal" robustness requirement.

## 13. Multi-Provider Architecture & DeepSeek Migration

> **Superseded.** This section describes the `local`/`openai`/`deepseek` provider switch as it existed at this point in the project. The later rewrite (§17 onward) removed multi-provider support entirely and consolidated on DeepSeek only — `src/generation/providers.py` and `LLM_PROVIDER` no longer exist in the current tree. Kept below as a historical record of why DeepSeek was chosen and what its API constraints are (still accurate), not as a description of the current config surface. See `src/evaluation/paid_calls_v1.py` and the root README for current reality.

Originally OpenAI-only; OpenAI credits ran out mid-project, forcing a pivot to local Ollama (`local`/`openai` switch). When the corpus scale-up was decided ([§14](#14-full-corpus-scale-up-prepared-paused)), a third provider, **DeepSeek**, was added specifically because local inference's 140–225s/call latency has no path to require.md's 10s gate, and OpenAI wasn't available — DeepSeek's `deepseek-v4-flash` is also drastically cheaper than the original OpenAI pass ($0.14/$0.28 per M input/output tokens vs. gpt-5.4-mini's $0.75/$4.50).

Three DeepSeek API constraints were confirmed by research (not assumed) before writing any integration code:
1. **No embeddings endpoint** (checked 3 independent sources) — `EMBEDDING_*` config stays pointed at local Ollama (BGE-M3/nomic-embed-text) even when `LLM_PROVIDER=deepseek`; `ollama serve` remains a dependency for the embedding role only.
2. **No strict `response_format: json_schema`** — only `text`/`json_object` are accepted (schema-strict mode exists only via tool-calling's strict mode, which has its own documented malformed-JSON bug). Solved with a new `SUPPORTS_STRICT_JSON_SCHEMA` capability flag: the non-strict path uses `json_object` plus an explicit schema description in the prompt, with new defensive response validation (`_validate_citation_json`) since the server no longer guarantees shape.
3. **Rate limits are concurrency-based, not RPM-based** — confirmed via DeepSeek's own docs (`api-docs.deepseek.com/zh-cn/quick_start/rate_limit`): `deepseek-v4-flash` allows 2500 concurrent requests, `deepseek-v4-pro` (the judge model, so the real bottleneck for RAGAS) allows 500. Judge pricing is ~4–7x flash depending on peak/off-peak (off-peak ≈ $0.66/$1.98 per M tokens), with peak roughly 2x off-peak — worth batching large eval runs off-peak.

Model roles under `deepseek`: generation = `deepseek-v4-flash`, judge = `deepseek-v4-pro` (kept on a different tier from the generator to preserve the same avoid-self-grading property the local provider has), rewrite = `deepseek-v4-flash`.

**Concurrency tuning (smoke-tested on a 10-item sample before any large run, per explicit instruction not to run code until approved):**

| MAX_WORKERS / RAGAS_TIMEOUT | TimeoutErrors | NaN metrics | RAGAS wall (8 scored) |
|---|---|---|---|
| 10 / 180 | 6 | 4/7 items | — |
| 10 / 400 | 6 | 4/7 items | 533.0s |
| **100 / 400** | **0** | **0/8** | **301.5s** |

10 concurrent workers was using under 2% of the real 500-concurrent judge-model ceiling — raising to 100 both eliminated every error and cut RAGAS wall time by ~1.77x.

## 14. Full-Corpus Scale-Up (prepared, paused)

A second, much larger corpus was prepared for an eventual scale-up: **488 documents, 661 dialogues, 4335 eval pairs** (`evaluation_category`: answer=3214, clarification=880, unanswerable=241), sourced from the original MultiDoc2Dial data. Design decisions made before any code was written:

- Clarification-type items are merged into the answerable pool — no new "ask a clarifying question" capability is being added to the pipeline.
- No synthetic multi-document composite questions for this corpus.
- No negative-document annotations required (unlike the subset's hard/medium/easy negatives).
- The 241 unanswerable items are trusted as pre-labeled by MultiDoc2Dial itself, not individually re-verified (explicit user decision — the subset's own 15 items *did* need correction, but this is 16x more items from a different, already-published source).

Preprocessing is complete: 488 docs cleaned, 1388 chunks materialized (`src/chunking/materialize.py`), embeddings computed with local BGE-M3 (159.9s for all 1388 chunks — throughput ~229ms/chunk regardless of corpus size, confirmed not a bottleneck at any realistic scale) and indexed into a separate Chroma collection (`data/processed/chroma_db_full`, kept fully isolated from the subset's `chroma_db`). A `src/data_prep/` package converts the new corpus's native `eval_turns.json` schema into this project's `test.json` schema, with a working stratified sampler (by category × domain × has_history).

**This run is currently paused, not abandoned.** After a clean 100-worker DeepSeek smoke test on a 10-item sample of the full corpus, the naive extrapolation to the full ~3275 RAGAS-eligible items was ~34 hours of unattended API-bound wall time. Rather than commit to that, the decision was made to first fully isolate the two tracks — `run_eval.py` now takes `--test-set`/`--chunks`/`--embeddings`/`--chroma-dir`/`--log-path`/`--out`, all defaulting to the original subset's paths so nothing changes by default — and to validate DeepSeek completely on the small, well-understood subset before returning to the large corpus. See [§15](#15-latest-result-full-deepseek-run-on-the-subset).

## 15. Latest Result: Full DeepSeek Run on the Subset

With isolation in place, the full 101-item subset was run end-to-end against `deepseek-v4-flash`/`deepseek-v4-pro` (writing to `generation_eval_results_deepseek.json` / `generation_deepseek.jsonl`, never touching the mistral-16k baseline or the original GPT-5.4-mini log entries):

- **Pipeline: 100/101 succeeded** in 166.9s. One item (`md2d-v2-va-001`, a routine 10-turn Chinese question, nothing unusual about it) failed with an empty-response `JSONDecodeError` — looks like a transient API flake rather than a content-specific issue; not yet given a retry-on-empty-response fix.
- **RAGAS: 86/86 eligible items scored**, 1 TimeoutError, 1 residual faithfulness NaN.
- Full metric table is in [§9](#9-generation-evaluation-ragas) above — this is the best-performing generation config measured in the project so far on 4 of 5 RAGAS metrics, with the important same-judge caveat noted there.

## 16. Open Items / Known Doc Gaps

- **`DESIGN_NOTE.md` is stale.** It still lists temperature sensitivity as "required, not yet run" (it has been, [§10](#10-temperature-sensitivity-requiremd-required-3-settings)) and doesn't mention the DeepSeek migration or its results at all. Needs a refresh before final submission.
- ~~**`providers.py`'s DeepSeek docstring is also stale**~~ — moot: `providers.py` no longer exists at all as of the later rewrite (§17 onward), superseded by `src/evaluation/paid_calls_v1.py`. See the correction note at the top of [§13](#13-multi-provider-architecture--deepseek-migration).
- **Same-judge rescoring** between mistral-16k/gpt-5.4-mini and deepseek-v4-flash would cleanly separate the generation-model effect from the judge-model effect on context_precision/recall (mirrors what `rescore_openai_baseline.py` already did once).
- **The 1 empty-response pipeline failure** on the DeepSeek subset run has no retry logic yet — worth adding regardless of whether the full-corpus run resumes, since a ~1% flake rate compounds to dozens of failures at thousands of items.
- **p90 latency still misses the ≤10s require.md gate** even on DeepSeek (60% ≤10s, not 90%) — not yet root-caused.
- **Full-corpus run**: preprocessing done, concurrency validated clean at 100 workers, isolation code in place — paused pending a decision on whether/how to proceed (further concurrency tuning, checkpointed incremental writes given the ~34h estimated wall time, or a smaller intermediate batch first).
- **Prompt-injection hardening**: explicitly deferred to future work per product decision, not forgotten.


## 17. Frozen Evaluation v1.0: Measurement Before Optimization (2026-09-06)

The next experiment now has a fixed data contract: all **488 source documents**, **100 dev + 100 holdout targets**, and a **20-item smoke/calibration subset of dev**. Each full split contains 70 answer, 20 clarify and 10 authored capability-refusal cases. Source document/test hashes, exact IDs, reviewed references, dialogue assignments and metric denominators are published in [the v1.0 manifest](evaluation/v1/releases/v1.0/release_manifest.json).

This supersedes the earlier assumption in §14 that the 241 source `unanswerable` labels could be trusted directly. All 20 initially sampled `respond_no_solution` targets were unsuitable as refusal QA examples. The new refusal cases exercise inaccessible personal/live information (8 per split) and obvious out-of-domain requests (2 per split), without selecting cases by retrieval similarity. They are reported separately and do not establish performance on naturally distributed unanswered questions.

Review covered 321 source candidates plus 20 authored cases. A final content audit overturned 10 earlier approvals for translation, history/reference and missing-condition defects. The final 200 items comprise 113 initial candidates, 67 source replacements and 20 authored cases; 12 approved source items remain in reserve. **These are AI-agent review decisions, not independent human labels.** The legacy `manual_*` filenames must not be described as human review. Reference and selection bias remain limitations.

The split has zero dialogue overlap and excludes all 103 historically exposed dialogues recoverable from local records from holdout. Previously unassigned dialogues were allocated before item review using a seeded 25% dev / 75% holdout rule; 29 new dialogues entered final dev. Fixed histories test one-step multi-turn behavior, not closed-loop generated conversations. Shared source documents and related refusal templates mean this is not unseen-document or fully semantic-template-isolated generalization.

The release publisher now verifies original source bytes, candidate JSON/CSV hashes, original question/history/evidence preservation, IDs, dialogue leakage, exposure and label quotas before publishing into a new directory. It refuses overwrites; a read-only verifier and 10 offline regression tests cover corruption, leakage, incomplete review and deterministic reproduction. The source corpus inventory does **not** yet certify the lineage of existing full chunks, embeddings or Chroma.

No model API was called for this preparation and no new quality/latency score is claimed. Pro is an explicit project judge default, not a RAGAS default. The frozen protocol specifies cost-controlled Flash calibration as the next step, but the legacy judge configuration has not changed. The legacy evaluator now rejects frozen v1 files before creating the retriever/generator, because its scoring does not yet implement v1 clarification handling or all-submitted-item denominators. Next: implement the v1 generation/scoring runner with explicit judge settings, complete item statuses, checkpoint/cache/usage tracking; verify derived-corpus lineage; calibrate on fixed dev outputs, then establish the full-corpus baseline.

## 18. Full-Corpus Retrieval Re-evaluation (2026-09-06)

The local retrieval stage is now complete on the frozen **488-document corpus and 100-item dev split**. All main evidence-hit results below use the same 70 answer items; clarify/refuse diagnostics are separate. Holdout was not queried or used for selection. Plans, environment/model revisions, content fingerprints and per-item results remain under `data/experiments/eval_v1/retrieval_fixed/`. **No generation or RAGAS run was made; paid model API calls: 0.** Local GPU inference was used; the public tokenizer vocabulary was downloaded once.

First, verify the artifacts rather than assuming corruption. Fresh parsing and cleaning were byte-identical to all 488 prior full outputs. Rebuilding the original 1,388 chunks reproduced their IDs/order/text, and fresh BGE-M3 encoding reproduced the old `.npy` file hash with zero elementwise difference. This closes the derived-artifact uncertainty recorded in §17 for that batch. The old ID-only cache mechanism was nevertheless unsafe for future edits: it now validates ordered content, encoder configuration/revision and payload hashes, with verified per-text vector reuse. Chroma also checks stored text, metadata and vectors rather than only IDs. Standard legacy demo caches migrate on first initialization.

The evaluation uncovered real chunking defects: invalid UTF-8 boundaries in overlap/recursive hard splitting, lost sentence separators, and inherited lead text dropped during semantic recursion. After fixes, semantic 1024 produces 1,385 chunks; semantic 512 produces 2,482; recursive 512/overlap 64 produces 1,859. Evidence scoring now uses exact normalized source-block intervals, only answer-side annotations, and unions of actually retrieved fragments. Background-only hits, duplicated fragments and fuzzy amount/negation matches cannot count as complete answer evidence.

| Full-dev configuration | Answer evidence hit@10 | Mean retrieved tokens@10 |
|---|---:|---:|
| Semantic 1024 + BGE-M3 dense | **46/70 (65.7%)** | 5,566 |
| Semantic 512 + BGE-M3 dense | 41/70 (58.6%) | 3,104 |
| Recursive 512/64 + BGE-M3 dense | 42/70 (60.0%) | 3,939 |
| Semantic 1024 + truncated base reranker | 52/70 (74.3%) | 6,207 |
| Semantic 1024 + windowed base reranker | **53/70 (75.7%)** | 6,503 |

Equal-token comparisons qualify the fixed-k result: semantic 512 gains two answer hits over semantic 1024 at a 3,000-token budget; semantic 1024 and recursive tie at 46/70 under 6,000 tokens. BM25 and the two tested gated RRF weights do not beat corresponding dense baselines. A controlled BGE-M3/E5 comparison gives both models identical passage/query text and verifies zero input truncation: BGE-M3 scores 38/70 versus E5's 24/70 at top10, with E5 weaker on this dev's cross-language slice. That comparison uses a common shortened passage view and must not be compared directly with the main chunking table as a model-replacement gain.

The reranking conclusion from §6 changes under the new corpus, queries and scoring protocol. Dense top20 supplies all candidates. Both rerankers use the same recent query tail; the windowed version scores every passage window and takes the maximum per parent chunk without gold selection. Full-dev confirmation follows the fixed smoke screen: windowed top10 gains nine hits and loses two versus dense top10. Direct truncation already captures most of this top10 gain, with about 0.59 seconds extra p90 versus 1.17 seconds for windowing. Windowing adds only one top10 hit over truncation, so its complexity needs a cost/quality justification.

A more economical context candidate is **windowed top5: 45/70 hits and 3,568 mean tokens**, versus dense top10's 46/70 and 5,566 tokens. This is about 36% fewer retrieved tokens, not a measured 36% reduction in total API cost. Under the same 6,000-token budget, windowing instead improves 46/70 to 52/70. The next generation comparison therefore retains three configurations: dense top10, truncated reranking top10, and windowed reranking top5. Production remains dense pending that validation.

The actual Chroma Retriever was checked on all 100 dev queries: warm sequential query embedding plus top20 search has p50 about 36 ms and p90 about 64 ms; mean top10 set agreement with exact dense ranking is 99.5%, with the same 46/70 answer hits. This excludes rewrite, reranking, generation and judge time and is not an end-to-end latency acceptance result.

Remaining limits are explicit. These new hit rates are conservative annotation-coverage proxies, not final answer accuracy or RAGAS Context Precision, and cannot be read as a regression from the legacy 96.5% recall. Four answer cases have incomplete source-block availability under exact alignment, all from one Chinese scanned PDF, and remain in the denominator. Of the other 20 baseline misses, 17 fail to retrieve the answer document and three retrieve it without complete evidence. That motivates targeted query/document error analysis before another broad chunking grid. The similarity threshold of 0.55 also fails to filter eight domain-related capability-refusal queries; actual generated refusal behavior remains unmeasured.

Validation includes 28 offline regression tests for release integrity, cache/index invalidation, evidence accounting and chunk text preservation. The unchanged v1.0 release retains its original hash. **Still pending:** the v1 generation/scoring executor, explicit inexpensive judge configuration and calibration, actual cost/end-to-end latency measurement, and final holdout acceptance after configuration lock. The initial `retrieval_v1` folder is a partial diagnostic run; formal results are in `data/experiments/eval_v1/retrieval_fixed`.

## 19. Smoke Generation, Cost Controls and Judge Calibration Findings (2026-09-06)

The next stage added an executable runner separating offline planning, live sequential generation, concurrent Flash judging, Pro baseline comparison and reporting. Exact-input caches, content-verified checkpoints, per-attempt usage, explicit non-thinking mode, output limits and a shared 5 CNY reservation ledger replaced implicit provider/judge defaults. The retained machine results are under `data/experiments/eval_v1/generation_smoke_v1/`.

All three candidates used the same 20 smoke targets and generator prompt: **14 answer, 4 clarify, 2 refuse**, with all 488 source documents available. Generation used DeepSeek Flash, temperature 0 and a 1,024-token output cap. Both judge tiers used explicit prompts and a 2,048-token cap. Thinking was disabled on every request. The new prompt also explicitly distinguishes answer/clarify/refuse, so differences from the legacy subset cannot be attributed to a single parameter change.

| Candidate | Successful generation | Raw Flash correctness, all 20 cases | Successful-request p90 | Successful and ≤10s / all requests |
|---|---:|---:|---:|---:|
| Dense top10 | 19/20 | 12/20 | 2.59s | 19/20 |
| Truncated reranking top10 | 20/20 | 13/20 | 4.31s | 20/20 |
| Windowed reranking top5 | 19/20 | 15/20 | 3.97s | 19/20 |

These are **uncalibrated smoke proxies**, not final acceptance. The two failed generations remain incorrect in the denominator. Windowed top5 gains three correct cases without losing an originally correct case, but the exploratory paired interval spans 0 to +30 percentage points. Its mean returned context is 3,317 tokens versus dense's 5,267, and observed generation API spend is about 0.090 versus 0.140 CNY. This is a useful cost/quality candidate; production defaults have not changed. Timing is warm, concurrency 1, including actual retrieval/reranking/generation/retries; initialization and offline judges are separate.

The complete iteration cost is **3.0381 CNY estimated from provider usage**, covering **289 API attempts**: generation 0.3701, initial Flash judges 0.8161, Pro comparison 0.9086, three diagnostics 0.0064, and a separately versioned faithfulness revision 0.9368. All attempts have recorded usage; no unresolved billing reservation remains. Different sample sizes and metric suites mean this cannot be compared with the older ~30 CNY batch as a like-for-like percentage saving.

Calibration exposed more than configuration costs. Flash and Pro agree on correctness for all **19/19 valid baseline outputs**, but both rely on the same references. Source inspection confirmed that one frozen question (`58f62ac…::agent-turn-10`) has no education-credential antecedent in its supplied history, despite a reference demanding a homeschooling question. The same gap is present in the local original MultiDoc2Dial dialogue; earlier agent review overlooked it. A second broad revocation question has a clarification-versus-general-guidance rubric ambiguity requiring independent review. **No v1.0 item or raw score was silently changed**, and the generated answers to the defective question are not automatically relabeled correct.

The first faithfulness prompt also had an ambiguous claim boundary: both judges emitted a supported context-availability statement with no supporting chunk ID, triggering validation failure. A separate `faithfulness_v2` prompt explicitly excludes such meta statements from external factual claims and reuses the same generated answers. All 77 v2 judgments parse and validate, but a semantic counterexample remains: Flash still counts the excluded statement as unsupported (score 2/3), while Pro excludes it (score 1). The 18 claim-bearing baseline score pairs have mean absolute difference about 0.039 and maximum difference 0.333. **Schema success and close average scores do not certify semantic calibration.** Both versions are preserved, are explicitly custom proxies rather than RAGAS, and independent human review is still incomplete.

A generation diagnostic also returned only whitespace with `finish_reason=stop`. The retry ceiling prevents unlimited spending, but JSON mode is not a reliability guarantee. Original failures retain their error type, usage and response ID; bounded diagnostic tools capture raw final content without overwriting the experiment. The next implementation revision should capture failed final content uniformly and validate a targeted structured-output improvement on fixed inputs.

Verification now includes **41 passing offline regression tests**, real local execution of all 60 retrieval configurations, and isolated resume checks for both scoring versions with **zero new API calls, zero model reloads and an unchanged ledger**. Frozen v1.0 hashes and holdout isolation remain intact. Next priorities are a reviewed v1.1 data/rubric revision and independent judge calibration before expanding to dev, then formal RAGAS Context Precision, generation sensitivity/load checks and final holdout acceptance.

## 20. Versioned Data Corrections and Offline Calibration Pack (2026-09-06)

Published **evaluation v1.1**, preserving v1.0 and every previous model result. The two corrections address evaluation validity: the broken-history item now requires clarification of the unidentified referent instead of an unsupported homeschooling follow-up; the general revocation-information question is relabeled answer, with explicit conditions for acceptable general guidance and rejection of unsupported personal or branch-specific assertions. The ambiguous item's historical evidence is retained as provenance but removed from active reference evidence and retrieval targets. Its behavior remains in the correctness denominator; retrieval is not applicable. These are **agent-reviewed corrections made after inspecting dev outputs**, not independent human labels or pre-registered changes. The exact delta is stored in `evaluation/v1/releases/v1.1/revision.json`.

No questions were replaced or reordered. All 488 corpus documents, question/history inputs and holdout item payloads are unchanged. Corrected label counts are dev **71/19/10**, smoke **15/3/2**, holdout **70/20/10**. Preserving the sample is more defensible than replacing items to restore old quotas. The derived publisher pins the v1.0 manifest, records exact field changes and rejects overwrites; its verifier rejects arbitrary edits even if file hashes are recomputed. The v1.0-only runtime rejects the new contract, preventing accidental reuse of old scoring assumptions.

An offline audit verified **60/60 generation records** against the original plan, item hashes and full request messages/configuration. All **58 valid outputs and two failures** can be retained; none was regenerated, and original latency remains attributed to its original run. No old correctness score is promoted to v1.1. The calibration pack contains 60 blinded review entries with empty human fields, a separate private system mapping, six affected-response review notes and 116 unsent candidate metric inputs. The three answers to the broken-history question still fail to resolve the referent; fixing the reference does not automatically make them correct. Any score change caused by rescoring must be reported as an evaluation revision, separately from model improvement.

A candidate scoped faithfulness contract records both external facts and excluded context-availability/capability/non-assertion units with reasons. Twelve AI-authored contrast cases include Chinese/English negatives, mixed statements, personal-status hallucinations and missing conditions. Scope exclusions do not establish correctness: falsely saying the retrieved text lacks an address can still be wrong. No-claim outputs remain undefined, and unsupported external facts remain eligible. Structural tests explicitly demonstrate that incorrect semantic scope can pass schema validation; **independent semantic calibration remains incomplete**. This is a custom candidate metric, not a RAGAS result.

This stage incurred **zero additional paid API calls and zero additional CNY**. The full offline suite passes **51 tests** (10 new). The initial suite invocation omitted the existing tokenizer-cache environment and failed during an attempted vocabulary download; using the documented local cache resolved the environment issue without modifying chunking code. Next: independently review the fixed calibration material, test the candidate judge on scope/coverage counterexamples, then lock the rubric and rescore smoke before expanding to dev. Formal RAGAS Context Precision, sensitivity/load tests and untouched holdout execution remain pending.

## 21. Delivery Work: Human Review, Budgeted Scoring and Shared Runtime (2026-09-06)

At this stage, a readable 20-case review form (later removed from the submission) linked to complete question/history, answer and evidence pages, with system identity and prior judge scores hidden. Human fields remained empty. The importer checked input versions, case completeness, verdict/reason and claim-count consistency; it preserved pending/uncertain judgments and required reviewer/exposure metadata. It explicitly could not certify reviewer identity or independence. Prior exposure through this conversation had to be disclosed rather than erased by calling the export blind.

`src.evaluation.score_v11` consumes versioned generation bundles for smoke/dev/holdout, with the legacy import limited to the audited 60 smoke records. Planning and reporting are offline; judging is a separate command with a shared ledger, exact-input cache and immutable plan/checkpoint bindings. Two unexecuted plans are prepared: 116 correctness/faithfulness tasks (112 unique verdict inputs), and 45 answer-target Context Precision tasks requiring 375 per-context verdict inputs (201 unique). Each plan defaults to a 2 CNY execution cap; the much larger byte-token worst-case forecast is not predicted billing. All old generation failures remain visible; missing scores remain unknown.

The Context Precision adapter uses pinned RAGAS 0.3.1's unchanged prompt/examples and metric aggregation, with role-labelled history explicitly included in user_input. It replaces transport with the audited JSON caller, bypasses framework-level retries and the telemetry wrapper, and disallows extra repair calls. Controlled offline outputs exercise the actual RAGAS metric, including [1,0,1] producing about 0.8333 and a mid-metric failure remaining unknown. No remote judge has been run through the new adapter yet.

The new `src.service.demo` and `src.service.evaluate` share `RagRuntime.answer`, with the same candidate retrieval, generation prompt, budgeted client and output guard. Only question/history/session metadata crosses into generation; gold is excluded. The runtime supports isolated/resettable in-memory sessions, explicit history limits, bilingual low/no-retrieval refusal, citation-ID failure handling and request traces. Retrieval contexts are retained separately from generation contexts, including gated/failed requests, so retrieval evaluation need not be conditioned on generator success. Runtime manifests link log IDs to configuration and artifact/code hashes. A complete progress bundle keeps not-submitted rows, and resume does not reload a model or regenerate completed failures.

PII handling now covers Chinese-adjacent phones and 18-digit IDs with X/x, phone formatting, SSNs, email and nested log values/credential fields. Regression checks preserve dates, amounts, form IDs and hexadecimal trace hashes. This remains limited pattern-based log protection, not a claim that outbound model data is fully anonymized. The runtime also avoids copying arbitrary SDK exception text into shared request logs. Citation-ID validity remains distinct from semantic support; a pure clarification can have no citations.

The full suite now passes **75 offline tests**, and the demo/artifact preflight plus a 20-question shared-runtime generation plan succeed locally. **No model was loaded by these preflights, no paid API call was made, and no new quality/latency result is claimed.** Earlier smoke measurements cannot be promoted to acceptance evidence for the new service behavior. Human calibration, real model integration, HTTP interface, adversarial/multi-turn tests, sensitivity/load studies, final holdout and consolidated deliverables remained outstanding at this stage. The human-review prerequisite was subsequently removed in §22.

## 22. Original Annotation Audit and Removal of the Human Review Gate (2026-09-06)

The user declined human blind review and asked to consult the original data. Human calibration is not required by `require.md`; making it a blocking delivery task had expanded the case scope. The active workflow now uses original-source provenance checks, the frozen case rubric and explicit automated judges, with limitations disclosed. Human forms remain blank, optional archives. Neither an AI review nor agreement between models is relabeled as independent human calibration. No frozen data, scoring criteria, historical outputs or budgets were changed by this operational decision.

A new read-only audit traces the dev set through the full multilingual conversion to the local original `multidoc2dial_dial_test.json` and `multidoc2dial_doc.json`. All **488** corpus document identities/converted metadata/frozen byte hashes match. All **90** source-derived dev questions preserve the original English question, answer, history, dialogue act and reference identity; **265** reference spans match original text and character offsets. The **20** smoke items are exact dev subsets. These checks certify lineage, not translation/OCR semantics or the correctness of every span-to-block alignment. No holdout questions were inspected for this source analysis.

The other **10** dev items are agent-authored refusal cases. Among the 90 source-derived items, **64** current reference answers differ textually from the transformed original answer, and **21** task labels change the conversion's clarification category to answer. These are documented case adaptations, not 64 proven source errors or untouched original gold labels. The resulting evaluation must be described as a MultiDoc2Dial-derived case benchmark, not an official MultiDoc2Dial score. Original reference answers remain preserved alongside adapted references.

The two disputed cases were traced to actual original turns and document spans. The ambiguous “neither” question already follows three “hi” turns in the local original; hidden credential references do not supply a visible conversational antecedent. In contrast, the DMV revocation dialogue genuinely asks whether an order was received, then explains revocation after the user says yes. v1.1's preference for a general answer is an application-task choice, not evidence that the original clarification annotation is wrong. These decisions were made after dev outputs had been seen and cannot be credited as generator improvements.

The delivery checklist, README and evaluation entry documents now remove the request for user labels. Next comes the already planned small-budget Flash scoring diagnostic, smoke rescoring and real shared-runtime integration, without waiting for a human form. Final tables can compare automatic estimates against target thresholds while explicitly stating their lack of independent human calibration. This audit made **0 paid calls**, incurred **0 new API cost**, and produced no new quality score.

## 23. Budgeted v1.1 Scoring and Live Runtime Smoke (2026-09-06)

Completed the accepted next experiment: four initial Flash judge tasks, the remaining frozen smoke rescore, a one-question RAGAS transport probe followed by all planned Context Precision tasks, and two initial shared-service generation requests followed by the remaining 18. A real CLI session also answered an English question and Chinese follow-up using the actual prior generated answer as history, then reset. No judge, generator or dataset binding changed during the runs. The machine results remain under `data/experiments/eval_v11/delivery_stage02/`.

| Historical candidate, v1.1 rescored | Correct / all 20 | External-fact Faithfulness | RAGAS Context Precision / 15 answer cases | Historical successful p90 | Historical online CNY/1,000 submitted requests |
|---|---:|---:|---:|---:|---:|
| Dense top10 | 13/20 | 0.974 (18 scored) | 0.579 | 2.586 s | 7.00 |
| Truncated rerank top10 | 14/20 | 0.983 (18 scored; 1 unknown) | 0.635 | 4.310 s | 7.03 |
| Windowed rerank top5 | 16/20 | 0.986 (18 scored) | 0.690 | 3.974 s | 4.48 |

The table reuses 58 valid historical answers plus two original generation failures; new scores do not make their original latency a new service measurement. All three correctness increases versus v1.0 are the same revocation case's changed rubric, not generator improvements. Windowed top5 gains three cases and loses none versus dense under the common v1.1 rubric, but its paired bootstrap difference interval is 0–30 percentage points and its correctness Wilson interval is about 58.4%–91.9%. Context Precision still misses the 0.70 target. Keep dense as baseline and windowed top5 as the main next-stage candidate; do not declare final acceptance or attribute a joint top_k/reranker change solely to reranking.

All 116 correctness/faithfulness tasks reached a terminal state: 115 succeeded; one truncated-candidate faithfulness judgment failed validation twice and remains unknown. Its possible inclusion gives a 0.931–0.984 Faithfulness bound. Each candidate also has one no-external-fact response, whose faithfulness is undefined. Reference-conditioned Context Precision completed 45/45 metrics from 375 verdict inputs, using 201 real API calls and 174 exact-input cache reuses. No Pro call or implicit RAGAS repair call occurred. Scope splitting and reference coverage remain limitations of the automated judge, not grounds to relabel it human-calibrated.

The live shared dense runtime submitted 20 requests: 19 succeeded, and one long-history VA request failed JSON parsing on both permitted attempts. Warm in-process successful-request p50/p90 were 2.376/2.767 seconds; 19/20 were successful within 10 seconds. This includes retries and logging but excludes startup, external interface transport and load testing. New service answers have not been independently scored and must not inherit the historical 13/20 result. Its observed token cost was 0.0315389 CNY, or 1.58 CNY/1,000 submitted requests under the observed provider cache conditions; this is not a new architecture cost-saving claim. The separate two-turn CLI check cost 0.0107731 CNY and shares the same service budget.

Total incremental spend was **1.140771 CNY across 336 API attempts**: scoring 0.8598391, Context Precision 0.2386199, service plus CLI 0.0423120. Every attempt has usage, each independent ledger stayed below 2 CNY, and completed-run resume made no new model calls. The 75 offline regression tests pass. New outputs include a reproducible joined analysis, AI diagnostic notes, actual-session message lineage, five redacted sample logs and a completion audit; original frozen releases and 307 historical result files remain intact.

The next controlled work is matched-k reranker on/off and at least three temperatures, plus bounded fixes for necessary clarification, history-distracted retrieval and JSON output failures. Clarify accuracy is only 0/3, 1/3 and 1/3 despite high factual grounding. These diagnostics provide a concrete optimization hypothesis before full 100-item dev confirmation; holdout remains unexecuted. Interface/security/load checks and final deliverable consolidation remain tracked in the delivery checklist.

## 24. Matched-k Reranking and Three-temperature Sensitivity (2026-09-07)

Completed the six-arm preregistered smoke matrix through the unchanged shared RagRuntime: dense@5, dense@10, windowed@5 and windowed@10 at temperature 0, plus windowed@5 at 0.3 and 0.7. All arms use the same 20 v1.1 questions, 488-document BGE-M3 index, dense top20 candidate pool, full role-labelled history, generator prompt, 0.55 gate, citation checks and bounded retry policy. Question/arm order was shuffled with seed 20260907. The controlled retrieval module extends the profile registry while preserving the frozen ranking method; old source and results remain untouched. Actual request hashes verify question/history/context-only generation and the three non-thinking temperature settings. Full results remain under `data/experiments/eval_v11/sensitivity_20260907/`.

| Fresh shared-runtime arm | Correct /20 | Faithfulness | Context Precision /15 | Generation failures | Observed / standardized uncached peak CNY per 1,000 |
|---|---:|---:|---:|---:|---:|
| Dense@5, t=0 | 13 | 0.899 | 0.574 | 0 | 6.63 / 9.01 |
| Dense@10, t=0 | 13 | 0.952 (1 unknown) | 0.579 | 0 | 2.28 / 14.49 |
| Windowed@5, t=0 | 15 | 0.981 | 0.690 | 1 | 2.24 / 10.28 |
| Windowed@10, t=0 | 15 | 0.980 | 0.663 | 2 | 9.00 / 18.10 |
| Windowed@5, t=0.3 | 16 | 0.967 | 0.690 | 1 | 2.06 / 10.10 |
| Windowed@5, t=0.7 | 15 | 0.991 | 0.690 | 0 | 1.96 / 9.49 |

Same-k reranking gains two and loses zero cases at k=5, and gains three/loses one at k=10. Increasing k adds no net correct answers; windowed@10 introduces an extra JSON failure and roughly doubles generator context tokens. Its sole apparent gain is a Standard ID answer whose judge accepts replacement advice despite missing the reference's explicit flight/passport condition. Dense@5 and @10 have identical correctness labels but different exact evidence coverage (8/15 versus 10/15), so equal observed correctness does not establish general equivalence.

Temperature 0.3's sole apparent gain is a Board hearing response. Both t=0 and t=0.3 omit the reference's explicit good-cause condition, yet the automated judge penalizes one and accepts the other. The diagnostic preserves both outputs, references and original judgments, without substituting an AI review for independent human labels. Therefore the raw 16/20 does not justify claiming a reliable temperature improvement. Keep temperature 0 for the main windowed@5 candidate and dense@10 as the continuity baseline, with dense@5 a cheaper-token diagnostic option. Do not promote all six configurations to full dev or redefine gold to fit the outputs. CP remains below 0.70. A 15/20 Wilson interval is roughly 53.1–88.8%; single-generation smoke comparisons are exploratory, not acceptance.

The matrix produced 116 successful outputs and four retained failures from 120 service requests. Six requests used the deterministic low-similarity gate without generation; 114 model requests plus four retries produced 118 API attempts. The failures concentrate on one long-history VA question and one extended repayment question. Their JSONDecodeError/ValueError outcomes all have provider finish_reason=stop; invalid raw text is not retained, so the evidence does not establish truncation or a specific malformed schema. Better bounded/redacted diagnostics and a separately versioned clarification/history intervention are the next priorities. Warm successful p90 ranges from 3.229 to 4.413 seconds, with 90–100% of all requests successful within 10 seconds, excluding startup, HTTP and load testing.

All 232 correctness/faithfulness tasks terminated: 231 succeeded and one dense@10 faithfulness task failed validation twice. It remains unknown; its conservative Faithfulness interval is 0.902–0.954. Every arm has one no-external-fact output excluded from the faithfulness mean. All 90 Context Precision tasks succeeded, including answer cases where generation failed. The 600 verdict inputs reduce to 198 exact inputs, of which 196 are compatible with the prior frozen scoring run; provenance-checked imports plus within-run caching require only two new CP API calls. Original dates, fingerprints and costs are retained; imports are not new provider measurements.

Total incremental cost is **3.3183536 CNY across 337 API attempts**, within the independent generation/quality/CP caps of 2/3/1 CNY: generation 0.4835054, correctness+faithfulness 2.8276354, CP 0.0072128. Token usage, retries and supplier caching are explicit. Dense@10's unusually low observed fee tracks 98.9% cached input versus 32.4% for dense@5; standardized peak uncached prices reverse that apparent advantage. Index/model reuse incurs no new embedding API charge; local machine cost is not monetized.

All 81 offline tests pass with the local tokenizer cache configured. A command missing that environment variable failed tokenizer import; its log is retained and the correct invocation documented. The completion audit verifies all 120 request/log links, 527 API cache envelopes, 200 cross-plan input imports, 502 historical file hashes including 307 legacy results, both frozen releases, exact matched-k prefixes and temperature context identity, zero oversized reranker pairs, settled usage and deterministic analysis. Completed-run resume adds no calls and leaves ledger/bundle hashes unchanged. No holdout execution, independent human calibration or external API/load acceptance is claimed. The updated delivery TODO marks the sensitivity experiment complete and keeps targeted fixes, full dev, final holdout, service/security work and deliverable consolidation open.


## 25. Scheduled Full Dev Run and Evidence-loss Diagnosis (2026-09-07)

The user requested the complete 100-question dev run during Beijing's 12:00–14:00 window. The process preloaded local models before noon, made its first new API request at **12:00:05**, completed generation at **12:08:22**, and finished all scoring at **12:22:26**. The frozen v1.1 dev set and two previously selected arms were used: dense@10 and windowed@5, both Flash, temperature 0, non-thinking. Forty compatible smoke records, including one failure, were retained with source hashes; 160 new generations produced one additional failure. No generator prompt, retrieval index or behavior was changed mid-run. The plans and per-item results remain under `data/experiments/eval_v11/dev_noon_20260907/`.

Before the run, a condition-coverage judge prototype required literal reference quotations and failed three of ten probes. An isolated inspection identified valid source/criteria paraphrases being rejected. Judge 1.3 allows semantic requirement summaries while still validating response quotations and the consistency of component decisions. The same ten AI-designed probes then produced the expected decisions. These are diagnostic controls, not independent human calibration. All 26 startup diagnostic API attempts remain archived and cost 0.1850626 CNY under one 0.5 CNY cap. Correctness was uniformly rejudged with version 1.3; Faithfulness and reference-conditioned RAGAS CP retain their frozen v1.1 semantics.

| Full dev metric | Dense@10 baseline | Windowed@5, t=0 |
|---|---:|---:|
| Generation successful / planned | 99/100 | 99/100 |
| Automatic correct / all questions | 66/100 | 65/100 |
| Correctness judge unknown | 7 | 5 |
| Possible correctness range from unknowns | 66–73% | 65–70% |
| Faithfulness mean, 94 fact-bearing scores | 0.9205 | 0.9396 |
| Faithfulness unknown / no-fact | 2 / 3 | 2 / 3 |
| CP, all 71 answer cases | 0.5859 | 0.7120 |
| Fresh successful-request p90 | 3.246 s | 4.401 s |
| Fresh successful within 10 s / submitted | 79/80 | 80/80 |
| Fresh observed / normalized uncached peak CNY per 1,000 | 7.19 / 15.24 | 4.80 / 10.39 |

The candidate meets Faithfulness and CP point-estimate targets, but the current automatic accuracy does not reach 80%, even with all unknown scores counted correct. Across 90 fully known pairs, it gains two and loses four; the exploratory paired bootstrap interval spans −7.78 to +3.33 percentage points, with ten pairs unknown. Thus the larger dev set does not establish a correctness improvement from the selected candidate. Its higher CP and lower input cost coexist with slightly lower complete annotation-block coverage: 46/71 versus 47/71. This comparison changes both reranking and k; attribution still depends on the preceding matched-k experiment. Same-output smoke comparisons show no known correctness flips from judge 1.1 to 1.3; the expanded sample, not a generator revision, explains the different full-dev measurements.

The run also exposes the judge's limits. Of 198 correctness tasks, 186 succeeded and 12 failed validation; 194 of 198 Faithfulness tasks succeeded, with four unknown. Three retained diagnostic cases show inconsistent handling of clarification versus supported conditional guidance. For example, the survivor-pension response covers both spouse and child branches but is rejected merely for not asking which branch applies. A CDL response explicitly asks what help is needed, yet its verdict overlooks that sentence and applies a different status-confirmation requirement than to the baseline. Consequently, the 5/19 clarify score cannot be equated with fourteen proven generation errors. Original scores are unchanged, and no hand-adjusted accuracy is reported. A new, uniformly applied judge version and specific redacted validation reasons should precede further prompt optimization.

New bounded generation diagnostics associate raw-format failures with provider response IDs without changing responses or retries. Three fresh attempts returned only spaces, despite finish_reason=stop. Two belong to one failed baseline request; the candidate's first attempt for that question was blank but its permitted retry succeeded. The historical smoke failure lacks raw content and is not assigned the same cause. New logs include a redacted length-limited excerpt, JSON position and schema category; API headers, arbitrary SDK exceptions and credentials are excluded.

An offline source-to-index investigation identified a separate, concrete ingestion defect. OCR merged the Board-hearing rescheduling paragraph into a long Markdown heading. The text, including the good-cause condition, survives cleaning, but when an oversized parent causes individual leaf visits, the semantic chunker drops a heading-only node before adding its breadcrumb. A minimal reproduction confirms the lost fact. This is not repaired by giving the generator a stronger prompt: the evidence never entered its index. There are 177 missing heading-only nodes across the corpus, including navigation, so their count must not be described as 177 lost answers. The four scanned dev questions all come from one PDF; other scanned-case exact-evidence gaps also involve OCR/alignment differences and do not prove every underlying fact was lost. The frozen index remains unchanged; a separately versioned content-preservation fix and rebuild are next.

All 142 CP tasks completed without unknowns. The complete dev run made **1,161 new API attempts for 4.4044233 CNY**: generation 0.9589814, correctness 1.3706879, Faithfulness 1.3130545 and CP 0.7616995. Including startup diagnostics gives **1,187 attempts and 4.5894859 CNY**, below the combined 8 CNY cap and each independent cap. All usage is known and settled. Exact-input imports reused 37 Faithfulness and 164 CP cache entries, retaining their old dates and receipts rather than billing them as new measurements. Fresh latency excludes the reused smoke rows and is internal warm runtime timing, not HTTP/load acceptance; local machine cost is not monetized.

The completion audit validates 160 fresh request/log identities, generation inputs without gold, identical dense candidate pools, 1,325 cache envelopes, 201 cross-plan reuse proofs and 952 historical artifact hashes. Every new dev API attempt began within the authorized noon window. Completed-run resume made zero calls and preserved 748 plan/result/ledger file hashes. Analysis reproduces deterministically; 86 offline engineering tests passed before the run. Holdout remains unexecuted and optional human-review fields remain blank. The delivery checklist now records full-dev execution and reporting as complete while keeping judge consistency, the versioned chunking fix, bounded multi-turn improvements, final service/security/load checks and holdout acceptance open.

## 26. Final Freeze, Holdout and Delivery Acceptance (2026-09-07)

The evidence-loss issue from §25 was fixed in a new `semantic_section_v2` module while the original chunker and 1,385-chunk index remained intact. The generic fix emits a non-empty heading leaf when an oversized parent is traversed; it does not inspect questions, gold data, evidence identifiers or document names. The repaired corpus has **1,591 chunks** and represents 72 of 76 unique dev answer blocks, versus 71 previously. A paired local comparison over the same 71 dev answer questions, BGE-M3 revision, query policy and rankings increased full-block hits from **47 to 48** for dense@10 and **46 to 47** for windowed@5, with no losses. The recovered Board-hearing question now retrieves the two-week/good-cause paragraph and received a correct final-smoke answer. Four blocks remain unavailable because of OCR or conversion alignment. The new vector manifest accurately records a full 1,591-row CPU encoding; an attempted copy did not yield row-level reuse.

The folder-based runtime now reads the encoder revision and device from the embedding manifest. This prevents a subtle startup failure discovered during comparison: the generic retriever created a model with a different device identity, rejected valid cached vectors, and began an unplanned full-corpus re-embedding. That run was interrupted before the old manifest was removed. The final explicit-folder path verifies chunks, vector bytes, ordered input hashes, encoder identity and Chroma payload before serving.

Correctness v1.4 was evaluated on 12 fixed positive/negative probes intended to accept both direct clarification and supported conditional guidance. It matched only **6/12** expectations; two probes ended in terminal schema-validation failure and valid conditional responses were still rejected. The experiment cost an incremental 0.0665147 CNY beyond the prior shared diagnostic ledger and is retained as a negative result. It was not used for final scoring. Version 1.3 remains the protocol-continuity proxy, with its narrower diagnostic and lack of independent human calibration stated everywhere.

The final configuration was saved before the first holdout call at **13:16:49 CST** (backup plan mtime 13:16:24; the earlier narrative time 13:17 was corrected during repository review): 488 documents, semantic-section v2, BGE-M3, dense top20, BGE reranker with 384-token windows, top5 output, confidence threshold 0.55, Flash at temperature 0, thinking disabled, 1,024 output tokens and at most two attempts. Its fixed 20-question smoke had 20/20 successful generations, 15 correct with one judge unknown (75–80%), Faithfulness 0.974 and CP 0.691. This proved the final path and the targeted content recovery, while already showing that all quality targets were not established.

The one-time frozen holdout then submitted all **100 questions**: 98 generations succeeded and two failed. Final automated results are **57 correct**, 10 judge-unknown and two generation failures, giving a 57–67% correctness range; **Faithfulness 0.9182** over 94 claim-bearing responses, with three no-fact and one unknown, has a conservative 0.9085–0.9190 range; **Context Precision 0.6616** covers all 70 answer cases with no unknown. Accuracy and CP are below their 0.80/0.70 targets. Faithfulness exceeds 0.85. Successful-and-within-10-second requests are **98/100**, with successful p50/p90 of 3.186/4.257 seconds, so the warm runtime latency target is met.

Final holdout generation used 294,476 prompt and 15,537 completion tokens across 100 attempts and cost **0.4648593 CNY**, or 4.6486 CNY per 1,000 submitted requests at the observed off-peak/cache mix; the same usage at peak rates is about 9.30 CNY per 1,000. Correctness, Faithfulness and CP judging cost 0.5709359, 0.6387176 and 0.4686940 CNY. Total final holdout measurement cost was **2.1432068 CNY**, within four independent 1.5 CNY caps. All usage is known and settled.

A new thin HTTP API reuses the CLI/evaluation runtime and implements health, ask and session reset with bounded JSON bodies. Five actual closed-loop conversations covered English/Chinese reference resolution, correction, refusal-to-topic-switch, longer history and session isolation; all 12 turns completed. Four prompt-injection/private-status probes returned refusals with no obvious prompt leak. A supplied SSN and returned phone numbers were redacted in traces. Eight concurrent low-confidence HTTP requests all completed within 1.900 seconds; this verifies the refusal path, not concurrent generation capacity. Cold model/index/service construction took 11.324 seconds. The final suite passes **93 tests** including live loopback HTTP contract checks.

The delivery documents now use one configuration and one set of final numbers: the root README, a 380-word design note, evaluation summary, validation record, five PII-redacted sample logs and the freeze report. The central story is the optimization process and its controls: invalid evidence was corrected, a larger corpus forced retrieval re-evaluation, a cross-encoder truncation issue led to windowed scoring, a content-preservation bug produced a general fix and measurable gain, and a more ambitious judge revision was rejected when its diagnostics failed. The remaining work belongs to a new version: expert calibration for clarification scoring, OCR/source-alignment controls, retrieval/answer completeness improvements and an untouched evaluation sample. The exposed holdout cannot be used to tune this release.


## 27. GitHub repository cleanup and self-contained source data (2026-09-07)

After the owner backed up the development workspace, the active repository was cleaned for GitHub submission. The complete transformed dataset was copied into `data/corpus/`: 495 files, all 488 documents, source manifests, 661 dialogues and 4,335 candidate turns. Every copied byte matched the source; the frozen release remains 100 dev + 100 holdout with smoke as a dev subset. No source test text enters retrieval.

Dependency review removed 60 obsolete Python source files and four historical test files, along with approximately 600 MB of old caches, duplicate conversions/indexes and delivery archives. Shared versioned implementations still used by the measured runtime/judge were retained. Seventeen measured execution files and the final chunk/vector hashes remain unchanged. The HTTP default now points to the final index, malformed session IDs are rejected, and missing billing ledgers cannot be reported as zero cost.

The current 69-test suite passed in an independent directory containing only prospective Git files. Both frozen-clean-text reconstruction and full reconversion of all 488 original documents reproduced the exact final 1,591 chunks on this machine, without a paid model call. Source/model setup, ignored runtime state, line-ending protection and a GitHub CI workflow are documented in the root README. A local timing audit corrected the earlier narrative freeze time: the backup plan mtime is 13:16:24, the execution marker is 13:16:36 and the first provider reservation is 13:16:49; these file times are local evidence. The final metrics and retained failures have not been changed.

See `doc/VALIDATION.md` for the current validation scope. The repository itself is the submission; the separate ZIP workflow is retired.
