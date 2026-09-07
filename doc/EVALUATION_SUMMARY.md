# Evaluation Summary

Evaluation date: 2026-09-07. The frozen plan was saved before the first holdout call at 13:16:49 China Standard Time. Local backup file times and the corrected narrative are documented in the freeze report. The measured runtime, prompt, index and judges remain unchanged.

## Result

The final system meets the Faithfulness and warm request-latency targets. It does not meet the automated correctness or Context Precision targets.

| Final 100-question holdout target | Result | Status |
|---|---:|---|
| Overall accuracy ≥80% | 57 correct / 100; 10 judge-unknown; 2 generation failures; possible range 57%–67% | **Not met** |
| Faithfulness ≥0.85 | 0.918 over 94 claim-bearing responses; 3 no-fact; 1 unknown; conservative range 0.908–0.919 | **Met** |
| Context Precision ≥0.70 | 0.662 over all 70 answer questions; 0 unknown | **Not met** |
| Successful and ≤10 s / all requests ≥90% | 98/100; successful p50 3.186 s, p90 4.257 s; maximum submitted 5.003 s | **Met, warm runtime** |

The accuracy score is an uncalibrated LLM-judge estimate. Even if every unknown judgment were correct, the upper bound is 67%, so judge instability does not change the threshold conclusion. By task, the proxy recorded 44 correct among 70 answer questions, 3 among 20 clarification questions, and 10 among 10 refusal questions; answer tasks also contain the two generation failures. Clarification scoring is a known weak point, but it cannot explain the full gap.

## Data and protocol

The “internal knowledge base” is represented by all 488 documents in the local `multidoc2dial_full_multiformat_multilingual` conversion. It includes English, Simplified Chinese, mixed-language documents, Markdown, TXT, DOCX, text PDFs, and scanned PDFs. This is government-service material used as a surrogate; it is not company-internal content.

Evaluation release v1.1 freezes 100 development and 100 holdout questions. Each split has 70–71 answer questions, 19–20 clarification questions, and 10 authored capability-refusal questions. The release traces 90 questions per split to original MultiDoc2Dial dialogues and adds 10 refusal cases. Across the release work, 64 reference answers were corrected/rephrased and 21 behavior labels were adapted. These changes and their provenance are disclosed; results must not be described as official MultiDoc2Dial benchmark scores. Smoke is a fixed 20-question subset of dev, so it is never added to the denominator.

Generation sees only the user question and recorded role-labelled history. Gold answers, labels, source annotations, and acceptance criteria are excluded. Every item is checkpointed. Exact input, code, release, chunk, embedding, prompt, model, and configuration hashes are bound into plans. Generation failures count as incorrect; judge failures remain unknown; no-fact responses do not receive an automatic Faithfulness score; Context Precision keeps all answer questions, including empty retrieval as zero.

## Frozen system

The final runtime uses `semantic_section_v2` at a 1,024-token soft target with no overlap, 1,591 chunks, BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`, dense top-20 candidates, BGE reranker base over fixed 384-token passage windows, and five returned chunks. DeepSeek Flash runs at temperature 0, thinking disabled, and a 1,024-token output limit. The runtime applies a 0.55 low-confidence gate, a 45-second provider timeout, and at most two attempts. Exact frozen bindings are retained in the final holdout `plan.json` and the v1.1 release manifest.

## Optimization process

| Problem | Controlled evidence | Decision and limitation |
|---|---|---|
| The original 91-document/101-question track had incorrect denominators, fuzzy evidence matches, stale caches, and invalid citation handling. | Rebuilt protocol, source lineage, exact position-based evidence coverage, content/encoder/vector hashes, and failure-preserving denominators. | Retired old scores from acceptance. The new track uses 488 documents and two fixed 100-question splits. |
| Corpus growth could invalidate the earlier chunking, embedding, and retrieval choices. | Re-ran chunking, BGE-M3 vs multilingual E5, BM25, dense, hybrid, truncating rerank, and windowed rerank on the 488-document dev corpus. | Kept BGE-M3 and section-aware chunks. Windowed top-5 reduced context and reached the strongest dev Context Precision, while hybrid and truncated reranking were negative results. |
| Large semantic chunks exceeded the cross-encoder input limit. | Compared direct truncation with fixed 384-token windows over the same dense top-20 pool. In the complete dev generation run, dense@10 CP was 0.586 and windowed@5 was 0.712. | Selected windowed@5 despite a slower warm p90 (4.401 vs 3.246 s) because it crossed the dev CP target and used less generation context. |
| A scanned PDF’s operative paragraph was parsed as a heading and disappeared when its parent was split. | A generic heading-preservation fix increased index representation by one answer block. At identical retrieval settings, full-block hits rose from 47 to 48 for dense@10 and 46 to 47 for windowed@5, with zero paired losses. Final smoke retrieved the missing paragraph and answered its two-week/good-cause conditions correctly. | Adopted the versioned fix. Four answer blocks still have OCR or conversion-alignment gaps. The new build re-encoded all 1,591 vectors; its manifest truthfully records no row reuse. |
| Clarification responses exposed judge inconsistency. | Correctness v1.3 passed a narrow 10-case diagnostic but failed 12 tasks in the two-arm dev run. A broader v1.4 rubric matched only 6/12 diagnostic expectations and had two terminal probe failures. | Rejected v1.4 as a negative experiment. Retained v1.3 only for protocol continuity and reported its unknowns and lack of human calibration. |

## Development and final-smoke evidence

The full dev comparison used two configurations over all 100 questions each. Dense@10 recorded 66 correct with 7 unknown, Faithfulness 0.921, and CP 0.586. Windowed@5 recorded 65 correct with 5 unknown, Faithfulness 0.940, and CP 0.712. Both had 99/100 successful generations. The paired automatic correctness change was inconclusive, while CP and context size favored windowed@5.

After the chunk fix, final smoke produced 20/20 successful generations. Correctness was 15/20 with one judge-unknown (75%–80%), Faithfulness was 0.974, and CP was 0.691. This validated the final execution path but did not establish all thresholds; the holdout was therefore treated as measurement, not a success demonstration.

## Sensitivity analysis

The required top-k, reranker, and temperature sensitivity used the earlier 1,385-chunk full-corpus index and the same 20 dev questions. It is historical selection evidence, not a repaired-index acceptance score.

| Configuration | Correct / 20 | Faithfulness | CP | Warm p90 s | Success ≤10 s | Observed generation ¥/1,000* |
|---|---:|---:|---:|---:|---:|---:|
| dense@5, t=0 | 13 | 0.899 | 0.574 | 3.704 | 100% | 6.63 |
| dense@10, t=0 | 13 | 0.952 | 0.579 | 3.229 | 100% | 2.28 |
| windowed@5, t=0 | 15 | 0.981 | 0.690 | 3.761 | 95% | 2.24 |
| windowed@10, t=0 | 15 | 0.980 | 0.663 | 4.370 | 90% | 9.00 |
| windowed@5, t=0.3 | 16 | 0.967 | 0.690 | 4.010 | 95% | 2.06 |
| windowed@5, t=0.7 | 15 | 0.991 | 0.690 | 4.413 | 100% | 1.96 |

*Observed billing is distorted by provider cache state and time-of-day pricing. Standardized uncached peak estimates for these arms were ¥9.01, ¥14.49, ¥10.28, ¥18.10, ¥10.10, and ¥9.49 per 1,000 respectively. Temperature 0.3’s one-question gain was not promoted because the sample was small and the judge disputed clarification behavior; deterministic temperature 0 was retained.

## Cost and operations

Final holdout generation made 100 provider attempts, used 294,476 prompt and 15,537 completion tokens, and cost an estimated ¥0.4649 in the 12:00–14:00 low-price window. This is **¥4.65 per 1,000 submitted calls** at the observed token mix, retry rate, cache hits, and off-peak rate; the same recorded usage at peak rates is approximately ¥9.30 per 1,000. It excludes local CPU/MPS time, OCR, storage, embeddings, and reranker compute.

Final judging cost ¥0.5709 for correctness, ¥0.6387 for Faithfulness, and ¥0.4687 for Context Precision. Total final holdout generation plus judging was ¥2.1432. Judge cost is evaluation overhead, not online serving cost. The final smoke cost ¥0.3152 and the separate service acceptance cost ¥0.0700.

## Service, multi-turn, security, and logs

The final CLI and HTTP API share the exact runtime. HTTP exposes `GET /health`, `POST /ask`, and `POST /sessions/reset`, returning `answer`, `action`, citations with document/chunk/heading/source file, `request_id`, and `session_id`. Invalid/missing citations make the request fail; a valid ID is described as syntactic validation, not proof of semantic support.

Five closed-loop conversations produced 12/12 successful turns across five isolated sessions. They covered English and Chinese reference resolution, a user correction, a personal-status refusal followed by a topic switch, and four-turn history. Cold process/model/index startup was 11.324 s. Warm final holdout requests met the 10-second target. A separate four-worker HTTP load probe sent eight low-confidence out-of-domain requests; all were explanatory refusals within 1.900 s. This load result covers the refusal path, not concurrent generation throughput, because generation is intentionally serialized to protect the budget ledger.

Four final security probes covered system-prompt extraction, fake administrator/private status, user instructions claiming higher priority, and persona hijacking. All returned refusals, leaked no obvious system prompt text, and emitted no citations. These probes demonstrate only the required minimum defense. Retrieved text is framed as data, user/history/context cannot override the system prompt, and output citation IDs are checked. There is no policy classifier, sandbox, or comprehensive injection guarantee.

PII is redacted when logs are written: the probe SSN became `[SSN]` and phone numbers became `[PHONE]`. The external LLM still receives original user input; logging redaction is not data-loss prevention for outbound API calls. The five submission samples in [sample_logs.jsonl](sample_logs.jsonl) cover answer, clarification, refusal, a retained generation failure, and PII/private-status handling. They contain trace IDs, versions, latency, retrieved IDs, citations, usage, cache state, and bounded error categories without API keys, headers, or raw SDK exceptions.

## Material limitations and next work

- Accuracy and Context Precision remain below target. The largest measured weakness is clarification handling, followed by missing requested conditions and retrieval misses.
- Correctness has no expert-labeled calibration. A 30–50 item stratified human set should be the next paid-quality investment; it should include conditional guidance, direct clarification, missing-condition, contradiction, and branch-confusion cases.
- Four source-answer blocks remain absent after conversion because of OCR/alignment. Add page-image OCR confidence, raw-text preservation checks, and targeted review for scanned documents before changing retrieval.
- The holdout is now exposed and cannot be used for tuning. Any improvement requires a new version and a new untouched evaluation sample.
- Model aliases can change at the provider. Response model IDs and fingerprints are logged, while the local embedding and reranker revisions are pinned.

Machine-readable final evidence is at `data/experiments/eval_v12/final_holdout_20260907/analysis.json`; service acceptance is at `data/experiments/eval_v12/final_service_acceptance_20260907/summary.json`. Historical protocol, source audit, dev, and sensitivity details remain under `doc/evaluation/v1/`.
