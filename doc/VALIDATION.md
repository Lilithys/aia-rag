# Validation Record

Validation date: 2026-09-07

## Final status

The final delivery configuration is bound by the stored holdout plan and v1.1 release manifest. The 100-question holdout was generated once after that freeze. All generation and scoring plans completed with failures retained, all cost ledgers are settled, and the offline aggregate reproduces from stored results.

The final quantitative outcome is mixed:

- correctness: 57/100 correct, 10 judge-unknown, 2 generation failures; 57%–67%, below 80%;
- Faithfulness: 0.918, conservative range 0.908–0.919, above 0.85;
- Context Precision: 0.662 across all 70 answer cases, below 0.70;
- successful and within 10 seconds: 98/100 on a warm runtime, above 90%.

The detailed methodology and denominators are in [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md). Accuracy uses an uncalibrated automated judge, but its upper bound still remains below the target.

## Corpus and release checks

- All 488 converted document identities and frozen file hashes match the full-corpus inventory.
- The frozen evaluation release contains 100 dev and 100 holdout questions; the 20 smoke questions are exact dev subsets.
- All 90 source-derived dev items were traced to original MultiDoc2Dial questions, histories, answers, dialogue acts, references, and 265 evidence spans.
- The release discloses 10 authored refusal cases per split, 64 rewritten reference answers, and 21 adapted behavior labels.
- Holdout questions and labels were not used for configuration selection.

These checks establish lineage and version consistency. They do not turn the adapted release into an official MultiDoc2Dial benchmark or certify every translation/OCR decision.

## Retrieval artifact checks

- final chunks: 1,591;
- chunks SHA-256: `497c665f2750bc4c90621036a6ccc394701d4a1d636ec69a0fa78e4b71a0f29f`;
- BGE-M3 revision: `5617a9f61b028005a4858fdac845db406aefb181`;
- embedding manifest SHA-256: `1bc07c37e6cc79608bdee45626940f1e7d0b801d0ce34966b49f43fcd03b4ce4`;
- vector shape: 1,591 × 1,024, finite float32 values;
- Chroma verifies IDs, documents, metadata, dimensions, and stored vectors against the manifest before reuse;
- the folder retriever uses the device and model revision recorded by the embedding manifest, preventing an accidental full-corpus re-embedding at startup.

The heading-preservation regression proves that the old strategy loses an isolated heading when an oversized parent is split and v2 retains it. Dev retrieval comparison shows one complete-answer-block gain and zero paired losses for both dense@10 and windowed@5. Four source answer blocks remain unavailable because of OCR or source/conversion alignment gaps.

## Service checks

The CLI preflight validates the final folder without provider calls. The HTTP API was exercised over an actual localhost socket with `GET /health`, `POST /ask`, and `POST /sessions/reset`.

- five closed-loop sessions, 12/12 successful turns;
- English and Chinese follow-ups, user correction, refusal then topic switch, four-turn history, and cross-session isolation covered;
- four security probes all refused: prompt extraction, fake authority/private status, instruction priority, and persona hijack;
- the test SSN was absent from the trace and appeared as `[SSN]`; phone numbers appeared as `[PHONE]`;
- eight concurrent low-confidence HTTP requests returned explanatory refusals within 1.900 seconds;
- cold model/index/service construction took 11.324 seconds;
- concurrent-generation capacity was not established; the runtime serializes generation to keep model/cache/ledger state safe.

## Automated regression

Final command:

```bash
TIKTOKEN_CACHE_DIR=data/experiments/tokenizer_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m unittest discover -s tests
```

Result after repository cleanup: **69 tests passed in 2.391 seconds** in an independent directory containing only prospective Git files, using the single command above. That directory had no `.env.local`, project virtual environment, original Desktop data path, Chroma database or raw API cache. The interpreter came from the locally synchronized locked environment. The suite no longer depends on removed legacy runners, human-review templates or raw provider caches. The HTTP tests require localhost socket permission in a restricted sandbox. Provider calls are mocked.

## Repository cleanup and reconstruction

The complete transformed source dataset is included at `data/corpus/`: 495 files including all 488 documents. Every copied file was compared byte-for-byte with the external source; the frozen source inventory additionally verifies document and dataset hashes. Formal evaluation remains 100 dev + 100 holdout, with smoke as a 20-item dev subset.

`python -m src.verify` checks source/release consistency, all 488 frozen clean files, the final chunk/embedding hashes, ordered vector-content bindings and **17 measured execution files**. The measured generation, prompt, retrieval and scoring implementations remain byte-identical to the final run.

Two independent local rebuild modes completed successfully:

- frozen cleaned documents → 1,591 chunks, byte-identical to the final chunk SHA-256;
- all 488 bundled original documents → format conversion/OCR → cleaning → 1,591 chunks, again byte-identical on this machine.

The full dev-only retrieval comparison was rerun after cleanup: dense@10 47→48/71 and windowed@5 46→47/71, with one gain and zero losses in each profile. Exact baseline replay requires Apple MPS because that historical encoder manifest records MPS; the final service index records CPU.

A final runtime check in the independent Git-file directory rebuilt Chroma from the included vectors, loaded the pinned embedding/reranker models, and retrieved five chunks. A low-confidence query returned an explanatory refusal with a provider stub that fails on any API call. Model/index construction took 3.895 seconds on this machine with weights already cached; this is not a fresh model-download timing or a replacement for the original service acceptance result.

Both rebuild modes reused the already verified final vectors only after proving chunk-file equality and made zero paid API calls. Fresh OCR can differ on another system; the builder reports that difference and writes to a new scratch directory.

Pinned model cache checks found 10 BGE-M3 files and six reranker files, approximately 3.43 GB total. A public setup command downloads exactly these revisions on a fresh machine. The model-download branch was not re-downloaded during cleanup; existing files were verified offline.

The old delivery ZIP and build script have been removed. The repository is the submission; Chroma databases, fresh API caches, secrets and scratch rebuilds are ignored by Git. Corpus and frozen artifact bytes are protected from line-ending conversion. Per-question retrieval comparisons remain with the frozen experiment artifacts under `data/experiments/`.

## Cost and completion checks

Final holdout API cost was ¥2.1432068: generation ¥0.4648593, correctness ¥0.5709359, Faithfulness ¥0.6387176, and Context Precision ¥0.4686940. No usage record is unknown. Final smoke cost ¥0.3151913 and service acceptance cost ¥0.0699581. Each phase stayed within its independent cap.

The final generation bundle has 100 submitted rows, 98 successes, and two retained failures. Correctness has 98 tasks with 88 successful judgments and 10 terminal judge failures. Faithfulness has 98 tasks with 97 successful judgments and one terminal failure. Context Precision has 70/70 completed question-level metrics, using 346 unique provider verdict calls for 350 planned context positions because four exact inputs reused cache entries.

## Known limitations

- correctness and Context Precision fail their targets;
- correctness v1.3 lacks independent expert calibration and is unreliable on some clarification/conditional cases;
- PII redaction protects stored traces, while original user input still reaches the external provider;
- the load test covers the low-confidence refusal path, and warm holdout latency is single-request execution;
- scanned-document OCR and exact source alignment remain incomplete;
- provider model names are mutable aliases even though response model IDs/fingerprints are logged;
- the holdout is exposed and cannot support further tuning of this version.
