# Experiment artifacts

This directory contains the machine-readable evidence behind the optimization
story. It is not another input corpus and is not read as knowledge by the RAG
service. Only the final index under `eval_v12/delivery_candidate/` is used at
runtime; the remaining folders support reconstruction, comparisons, metric
aggregation and auditability.

The directory names follow the evaluation lineage:

- `eval_v1` is the first controlled 488-document baseline.
- `eval_v11` corresponds to evaluation release **v1.1**. The folder name omits
  the dot; it does not mean “version eleven.”
- `eval_v12` is the next candidate line, created after fixing the
  heading-preservation defect, and contains the final submission evidence.

They do not refer to document counts, question counts, or model versions.

## Inventory

| Directory | What it contains | Role in the submission |
|---|---|---|
| `eval_v1/retrieval_fixed/` | Frozen cleaned text for all 488 documents, the 1,385-chunk baseline, BGE-M3 vectors and initial retrieval analysis | Rebuild input and before/after retrieval baseline |
| `eval_v1/generation_smoke_v1/` | Early 20-question generation/faithfulness smoke results | Historical evidence that motivated later evaluation work |
| `eval_v11/source_audit/` | Checks connecting adapted evaluation items to the original MultiDoc2Dial turns and evidence spans | Dataset-lineage evidence |
| `eval_v11/sensitivity_20260907/` | Six controlled settings covering top-k, reranker behavior and temperatures 0, 0.3 and 0.7 | Required sensitivity analysis and configuration selection |
| `eval_v11/dev_noon_20260907/` | Full development comparison for the two selected configurations | Main pre-freeze development result; never final acceptance |
| `eval_v11/dev_condition_diagnostic_20260907_v13/` | Diagnostic controls for correctness judge v1.3 | Judge behavior evidence; not an independent human calibration set |
| `eval_v11/delivery_stage02/` | Pre-delivery runtime and completion checks | Historical delivery-stage evidence |
| `eval_v12/delivery_candidate/semantic_1024_v2/` | Final 1,591 chunks, BGE-M3 embeddings and manifests after the heading fix | **The index used by the CLI, HTTP service and final evaluation** |
| `eval_v12/final_smoke_20260907/` | Final 20-question smoke run using a dev subset | Gate before the holdout; not 20 additional evaluation questions |
| `eval_v12/final_holdout_20260907/` | One-time 100-question generation, correctness, Faithfulness and Context Precision results, including failures and cost ledgers | **Final quantitative result** |
| `eval_v12/final_service_acceptance_20260907/` | Multi-turn, security, PII and concurrent low-confidence-request summary | Final local service acceptance evidence |
| `eval_v12/judge_v14_diagnostic/` | Negative experiment for a proposed judge rubric that passed only 6/12 controls | Rejected approach retained to show the optimization decision |
| `tokenizer_cache/` | Public tokenizer vocabulary | Enables offline token accounting and tests |

## Common file types

| Name | Meaning |
|---|---|
| `plan.json` | Frozen inputs, configuration, limits and code/artifact hashes for a run |
| `bundle.json` or `results.json` | Per-question outputs or metric results; failed items remain present |
| `completion_manifest.json` | Completeness and resume/audit record |
| `usage.jsonl` or usage ledger files | Token use and measured API cost |
| `analysis.json` or `report.json` | Offline aggregation derived from the stored results |
| `embedding_manifest.json` | Encoder revision, device, dimensions and content-binding hashes for stored vectors |
| `chunks.jsonl` | Retrieval chunks with document and heading metadata |

New runs must go under the Git-ignored `data/local/` path or a newly versioned
experiment directory. They must not overwrite these frozen artifacts. Derived
Chroma databases and provider API caches are intentionally excluded from Git.

The headline results and denominators are explained in
[`doc/EVALUATION_SUMMARY.md`](../../doc/EVALUATION_SUMMARY.md). The chronological
decision record is in
[`doc/PROJECT_SUMMARY.md`](../../doc/PROJECT_SUMMARY.md).
