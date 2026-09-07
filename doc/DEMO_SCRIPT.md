# 7-minute Demo Script

## 0:00–0:45 — Frame the case

Open the root README and state: the 488-document MultiDoc2Dial-derived corpus is a stand-in for a bilingual internal knowledge base. The goal is a traceable QA service and an evidence-led optimization process. The final system meets Faithfulness and warm latency targets; it misses correctness and Context Precision.

## 0:45–1:30 — Show the architecture and preflight

Run:

```bash
uv run python -m src.service.final_demo
```

Point out the frozen chunk and embedding hashes, pinned BGE-M3 revision, windowed top-5 retriever, Flash temperature 0, and disabled thinking. Explain that preflight performs no API call.

## 1:30–3:00 — Demonstrate grounded multi-turn QA

Start the CLI with the README command. Ask:

```text
How can I request a Board Appeal?
Can I reschedule its hearing?
```

Show that the second turn resolves “its,” returns the two-week/good-cause conditions, and cites the Board-hearing chunks. Explain the optimization story: OCR put the operative paragraph in a heading; the old chunker dropped the heading-only leaf; the versioned fix restored it and added one complete evidence hit with no paired dev loss.

Reset and ask:

```text
我想申请联邦学生贷款，需要先做什么？
```

Use the Chinese response and English source citation to show cross-language retrieval.

## 3:00–3:45 — Show capability boundaries

Ask:

```text
What is my current Social Security application status?
```

Show the refusal of personal/live record access and any supported static contact guidance. Mention that the low-confidence path can refuse before calling the model. The final security probes also refused prompt extraction, fake authority, instruction-priority, and persona-hijack attempts.

## 3:45–4:30 — Show the service contract and traces

Open `src/service/http_api.py` or call `/health`. Explain `POST /ask`, returned `session_id`/`request_id`, and `/sessions/reset`. Open one row in `doc/sample_logs.jsonl`: show config/run IDs, stage latency, retrieved IDs, citation check, token usage, retry/error category, and `[SSN]`/`[PHONE]` redaction. State that original user text still reaches the external provider; logging redaction is not outbound anonymization.

## 4:30–6:15 — Walk through the optimization evidence

Use the tables in `doc/EVALUATION_SUMMARY.md`:

1. Explain why the old 91-document scores were retired: bad denominators, fuzzy evidence, stale-cache risk, and citation gaps.
2. Show the 488-document re-evaluation. BGE-M3 survived; hybrid retrieval was a negative result. Windowed reranking solved cross-encoder truncation and moved full-dev CP from 0.586 to 0.712 versus dense@10, with a latency/context trade-off.
3. Show the six-arm sensitivity table. Temperature 0.3 gained one automatic score, but the change was a judge-disputed clarification case, so temperature 0 stayed frozen.
4. Show judge v1.4 as a rejected experiment: only 6/12 diagnostics matched. This prevented a rubric change from being presented as a model improvement.

## 6:15–7:00 — Present final results and next investment

Show the holdout table: correctness 57%–67% and CP 0.662 miss; Faithfulness 0.918 and 98/100 under 10 seconds pass. Online generation cost was ¥0.4649 for 100 calls, or ¥4.65/1,000 at the observed off-peak/cache mix. Judging cost is reported separately.

Close with three concrete limits:

- the holdout is now exposed and cannot be used for tuning;
- clarification correctness lacks expert calibration and remains the largest measured weakness;
- four annotated blocks remain unavailable after OCR/alignment.

The next investment is a 30–50 item expert-labeled judge calibration set, then a versioned clarification/OCR intervention evaluated on a new untouched sample.
