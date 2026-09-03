# aia-rag

RAG + Generative AI QA system built on the document set in `doc/` (see
`require.md` for the case-study requirements).

## Demo

```bash
uv run python -m src.demo
```

Interactive multi-turn CLI — `conversation_history` grows across every
question you ask in one run (`exit`/`quit`/Ctrl-D to leave), which is what
actually demonstrates require.md's "conversation continuity within the same
session" requirement, not just a one-shot question/answer script. Every
call is logged (`data/logs/generation.jsonl`, PII-redacted) under one
`session_id` per run, so a full conversation can be traced back as a unit.

Runs against whichever provider `LLM_PROVIDER` is set to (`src/generation/providers.py`;
`local` by default, needs `ollama serve` running with `mistral-16k` pulled
and no API key at all). To use OpenAI instead, set `LLM_PROVIDER=openai`
and `OPENAI_API_KEY` yourself before running — the demo has no API-key
handling of its own, by design; it just reads whatever's already in the
environment, the same as the rest of this project's scripts.

## Step 1: Preprocessing (PDF / DOCX / TXT / Markdown -> Markdown)

`doc/documents/` contains 91 source documents across 5 formats: native
Markdown, TXT, DOCX, text-layer PDF, and image-only (scanned) PDF. Before
anything else can be indexed, every document is normalized to Markdown with
its heading hierarchy preserved, since headings are what downstream chunking
will use as retrieval-friendly section boundaries.

### Setup

Requires [uv](https://docs.astral.sh/uv/) and Homebrew's `tesseract` OCR
engine with the Chinese Simplified language pack (documents are English,
Chinese, and mixed):

```bash
brew install tesseract
curl -fsSL -o "$(brew --prefix tesseract)/share/tessdata/chi_sim.traineddata" \
  https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata
uv sync
```

### Run

```bash
uv run python -m src.preprocess.run
```

Converts all 91 documents into `data/processed/markdown/*.md` (one file per
document, named after the source file) and writes
`data/processed/preprocessing_report.json` with per-document word/heading
counts and any failures.

Useful flags:

```bash
# only re-run one format
uv run python -m src.preprocess.run --formats pdf_scanned

# only re-run specific documents (by doc_id from the manifest)
uv run python -m src.preprocess.run --doc-ids "How to renew a license#1_0"
```

### How each format is converted (`src/preprocess/`)

| Source format | Module | Approach |
|---|---|---|
| Markdown | `convert_markdown.py` | Passthrough (already has headings); old frontmatter is replaced with a normalized schema. |
| TXT | `convert_txt.py` | No markup at all. First line = title (H1). A standalone line that doesn't end like a sentence is treated as a heading (H2, flat — plain text has no nesting signal). |
| DOCX | `convert_docx.py` | Reads paragraph styles directly (`Title`/`Document Title` -> H1, `Heading N` -> H(N+1)), plus lists and tables, in document order. |
| Text-layer PDF | `convert_pdf_text.py` | `pymupdf4llm`, which infers heading levels from font size/boldness in the embedded text layer. |
| Scanned PDF (no text layer) | `convert_pdf_scanned.py` | Renders each page to an image and OCRs it with Tesseract (`eng`/`chi_sim`/both, picked per document). Heading level is inferred from each line's visual salience — color departing from the page's grayscale body text, and height relative to the tallest heading line on the document — since scanned pages carry no font metadata. |

Every output file gets normalized YAML frontmatter (`doc_id`, `domain`,
`title`, `language`, `format`, `source_file`, `conversion_method`) sourced
from `doc/manifests/document_manifest.json`, so downstream chunking/indexing
can trace every chunk back to its source document.

**Known limitation:** heading depth recovered from TXT and scanned PDFs is a
best-effort visual/structural heuristic, not a guarantee of the original
semantic nesting — plain text and OCR output carry no reliable signal for
heading depth beyond 2-3 tiers. Native Markdown and DOCX (which carry either
literal Markdown syntax or Word paragraph styles) are exact.

## Step 2: Clean, chunk, and compare chunking strategies

Three sub-steps, each independently runnable (`src/chunking/`):

```bash
uv run python -m src.chunking.prepare          # clean the step-1 Markdown
uv run python -m src.chunking.run_experiment   # sweep strategy x size x overlap, compare
uv run python -m src.chunking.materialize      # write the final chunk set for indexing
```

**Cleaning** (`prepare.py` / `clean.py`) reads `data/processed/markdown/`,
drops leaked CMS/UI boilerplate sections (e.g. "Show 'do it online' button in
megamenu:", "相关PDF文件:" followed by bare filenames — backend field labels
with no answerable content), unescapes stray `\[` `\]` `\$` `\*` left by PDF
text extraction, and normalizes whitespace/unicode. Output:
`data/interim/markdown_clean/*.md` + `data/interim/cleaning_report.json`.

**Three chunking strategies** (`src/chunking/strategies/`), deliberately
built to differ in more than just size:

| | Strategy | Behavior |
|---|---|---|
| A | `fixed_size.py` | Token-window slicing, structure-blind. Baseline. |
| B | `recursive_structural.py` | Splits/packs at heading -> paragraph -> sentence boundaries to hit a *consistent target size*. Heading lines ride along as real content, not just metadata. |
| C | `semantic_section.py` | Keeps whole heading sections together (size is a soft cap, not a target); small subsections merge into their parent instead of fragmenting. Every chunk is prefixed with its heading breadcrumb *inside the chunk text* (`[Doc > Section > Subsection]`), which measurably helps retrieval on this corpus. |

**Comparison** (`run_experiment.py`, `evaluate.py`) sweeps
`{A,B,C} x {256,512,1024} x {0%,15%,30% overlap}` = 27 configs and scores
each three ways: evidence-span containment against the test set's own gold
answer spans (no retriever needed), BM25 retrieval recall/MRR/context-precision
(a lightweight, embedding-model-agnostic proxy — the eventual embedding model
is evaluated separately once chosen), and structural cost proxies (chunk
count, avg tokens/chunk). Results: `data/processed/chunking_experiment_results.{json,csv}`.

**Result: semantic/section-aware chunking (C)** at target size 1024 /
overlap 0 wins — best MRR of all 27 configs, competitive recall at ~35% fewer
tokens per chunk than fixed-size, and section-aligned chunks double as clean
citation units. B is the pick instead if 100% evidence-completeness matters
more than cost (it never split a gold answer span across chunks). Full
methodology, charts, and all 27 results:
**[Chunking Strategy Comparison](https://claude.ai/code/artifact/ef88843f-0ac7-433b-9df5-3e7fadb5d741)**.

`materialize.py` writes the chosen config's chunks to
`data/processed/chunks/chunks.jsonl` (one JSON object per chunk: `chunk_id`,
`doc_id`, `heading_path`, `text`, `token_count`, ...) — the input the next
step (embedding/indexing) will consume.

## Step 3: Compare embedding models

```bash
uv run python -m src.embedding.run_experiment --models bge_m3 multilingual_e5
```

Embeds the 199 chunks from step 2 with each model (`src/embedding/models/`:
`bge_m3.py`, `e5_multilingual.py`, `openai_embed.py` — each applies its own
documented query/passage convention, e.g. e5's required `"query: "`/`"passage: "`
prefixes) and re-runs the *same* evidence-matching + retrieval harness from
step 2 (`src/chunking/evaluate.py`), swapping BM25 for cosine similarity over
each model's embeddings — the embedding model is the only variable that
changes. Embeddings are cached per-model under `data/processed/embeddings/`.
Results: `data/processed/embedding_experiment_results.json`, including a
breakdown by `language_relation` (aligned / mixed-language / cross-lingual).

**Result: BGE-M3** beats both multilingual-e5-large and the BM25 baseline on
every overall metric (recall@5 85.9% vs 67.1% vs 69.4%; MRR@10 0.70 vs 0.59 vs
0.60) — but the real story is the breakdown: on the 18 test questions where
the query language differs from every required document's language, BGE-M3
gets the right chunk into the top 5 **67%** of the time; multilingual-e5-large
manages **6%**, barely above BM25's **0%**. multilingual-e5-large is
essentially tied with BGE-M3 on same-language questions and embeds ~1.7x
faster — it just isn't meaningfully cross-lingual for this EN/ZH corpus.
OpenAI's `text-embedding-3-large` is implemented in the same pipeline but
wasn't run (no API key this round); its per-token price applied to this
corpus's actual token counts puts it at roughly $0.014 one-time to index and
$0.025 per 1,000 queries, no API call needed to estimate that. Full
methodology, charts, and the qualitative cross-lingual example:
**[Embedding Model Comparison](https://claude.ai/code/artifact/b13c234a-76db-4880-88cb-13210938a643)**.

## Step 4: Build and compare retrievers

```bash
uv run python -m src.retrieval.run_experiment
```

Three-stage comparison (`src/retrieval/`), all on the same evidence-matching
harness from steps 2-3:

1. **Dense vs. sparse vs. hybrid.** Dense = BGE-M3 embeddings in a persistent
   Chroma collection (`chroma_store.py`, `data/processed/chroma_db/`) — 199
   chunks doesn't need a server or real ANN indexing, but Chroma gives actual
   on-disk persistence and metadata filtering for free. Sparse = the BM25
   baseline carried over from step 2. Hybrid = Reciprocal Rank Fusion
   (`indexes.py`, `score = Σ 1/(60 + rank)` per ranker — the standard,
   weight-free fusion method).
2. **Reranker on/off** on whichever strategy won stage 1, using
   `BAAI/bge-reranker-base` over the top 30 candidates (`reranker.py`).
3. **top_k sensitivity sweep** (k=1,3,5,10,20) on the final config — the
   analysis require.md asks for directly.

**Result: dense retrieval alone, no reranker.** Both things "expected" to
help — hybrid fusion and reranking — hurt instead:

- **Hybrid underperforms dense** (recall@5 69.4% vs. dense's 85.9%, MRR@10
  0.64 vs. 0.70) because RRF weights BM25 equally regardless of whether it's
  actually useful for a given query — and BM25 scores 0% recall on
  cross-lingual questions (step 3's finding), so on ~a third of this test set
  hybrid is averaging a good ranking with a near-random one.
- **bge-reranker-base cuts recall@5 to 70.6%** (from 85.9%) while being 17x
  slower (66ms → 1,110ms/query) — and it isn't the cross-lingual cases that
  suffer (those improve slightly); it's the same-language majority.
  `bge-reranker-large` doesn't hurt quality the way base did (recall matches
  dense-alone almost exactly), but at ~4s/query it would burn most of the
  10-second end-to-end latency budget on retrieval alone. Checked both sizes
  before concluding "skip it" wasn't just base being underpowered.

`data/processed/retrieval_experiment_results.json` has full numbers
including the top_k sweep (recommend **k=10**: recall 96.5% vs. 85.9% at
k=5, and where multi-doc recall crosses 94% — final call belongs with the
generation step once real latency is on the table). Full write-up, charts,
and the concrete "reranker demotes the correct answer" example:
**[Retriever Comparison](https://claude.ai/code/artifact/4460e5af-dcf3-4eed-b146-d74bfcb16e73)**.

`src/retrieval/retriever.py::get_default_retriever()` wraps the winning
config (dense, Chroma + BGE-M3, `top_k=10`, no reranker) for the next step
(generation) to import directly:

```python
from src.retrieval.retriever import get_default_retriever
retriever = get_default_retriever()
chunks = retriever.search("some question", top_k=10)
```

## Step 5: Generation (grounded QA, citations, refusal, logging)

```bash
uv run python -m src.generation.run_eval
```

`src/generation/` wires retrieval into an LLM call and back out to a
verified, logged answer:

- **No free-form chain-of-thought** — Faithfulness is enforced structurally
  instead: `llm.py` requires a single structured-output call (`answer`,
  `citations`, `sufficient_context`) rather than reasoning out loud, keeping
  latency and cost down. This turned out to matter more than expected: an
  early local-model choice (`qwen3.5:4b`) had "thinking" always on regardless
  of this design, and its hidden reasoning tokens sometimes ate the whole
  completion budget before reaching the answer — see the provider note below.
- **Confidence gate** (`pipeline.py`, threshold 0.55) — a cheap pre-filter on
  retrieval's top-1 similarity score, catching only the most obviously
  off-topic queries before spending an LLM call. Deliberately conservative:
  answerable/unanswerable score distributions on this corpus overlap too
  much for a single threshold to safely do more (see the step-4 discussion).
  The real answerable/unanswerable judgment happens inside the LLM call,
  which sees the actual retrieved text.
- **Citation guardrail** — every `chunk_id` the model cites is checked
  against what was actually retrieved, and anything that doesn't match is
  stripped before the answer is surfaced. This catches fabricated *source
  references* but not fabricated *content* — the answer text itself still
  ships even when every one of its citations gets stripped. On the smaller
  local generator this happens often enough (§ below) that it's flagged as
  a next step, not treated as solved.
- **Minimal prompt-injection defense** — the system prompt explicitly frames
  retrieved context as data to read, never instructions to follow.
- **Language matching** (`prompt.py::detect_language_hint`) — the corpus is
  only ever English/Chinese/mixed, so a cheap script-composition heuristic
  states the target language explicitly rather than trusting the model to
  infer it; measurably necessary (`gpt-5.4-mini` answered a plain-English
  question in Spanish on 2 of 3 runs before this fix, despite zero Spanish
  content anywhere in the retrieved context).
- **PII-redacted structured logging** (`logging_utils.py`, `pii.py`) — one
  JSONL line per query (`data/logs/generation.jsonl`, gitignored) with
  retrieval scores, refusal reason, citation validity, latency, and token
  usage; emails/phones/SSNs/ID numbers are regex-redacted before anything
  is written.
- **Provider-switchable models** (`src/generation/providers.py`) — OpenAI
  credits ran out mid-project, forcing a pivot to local models (Ollama).
  Rather than replace the OpenAI path in place, both configs live behind one
  `LLM_PROVIDER` env var (`local` default, or `openai`), so switching back
  needs no code changes. Getting the local path actually working surfaced
  two real Ollama bugs (its OpenAI-compatible endpoint silently truncates to
  a 4096-token context and silently ignores the documented override) —
  worked around with custom Modelfiles baking in a larger context window.
  Full debugging writeup in that module's docstring.

**Result on the full 100-question test set, current run** (generator:
`mistral-16k`, local; judge: `qwen2.5-16k` via RAGAS's standard metrics,
also local — **$0 API cost**): all four of require.md's numeric bars are
missed — **Faithfulness 72.0%** (target ≥85%), **Accuracy 25.0%** (target
≥80%), **Context Precision 56.7%** (target ≥70%), **latency p90 183.7s**
(target 90% under 10s). Latency is an accepted, deliberate trade-off for
running on local compute (see the provider note above), not a bug. The
Context Precision miss is mostly a *metric-definition* story, not a quality
one — RAGAS's version is rank-weighted (Average Precision), reads very
differently from the first run's flat hand-built ratio on identical
retrieval, and is independent of which model generates the answer (full
explanation in the report). Faithfulness and Accuracy, though, trace to a
real, quantified cause: `mistral-16k` fabricates or mis-copies a cited
`chunk_id` on **49.5% of non-refused answers** — checked against the raw
model output before the citation guardrail strips it — and those cases
correlate directly with wrong factual claims in the same answer (e.g. a
gold answer of "$200" generated as "$500"). Full root-cause writeup, real
examples, sample logs, and next steps: **[Generation Evaluation](https://claude.ai/code/artifact/17dc2bb1-8253-414b-865d-0fc7213996c1)**.

**First run for comparison** (generator: `gpt-5.4-mini`; judge: a hand-built
rubric, not RAGAS — metric definitions aren't comparable 1:1 with the
current run): Faithfulness 94.7% (pass), latency p90 3.1s (pass), Accuracy
68.4% and Context Precision 16.1% (both missed, both diagnosed — see the
report for detail). Real measured cost was $5.30 per 1,000 calls; this
run's own token counts priced at the same rate would be **$6.99 per 1,000
calls** (never actually billed — this run made zero OpenAI calls).

**Same-judge comparison** (`uv run python -m src.generation.rescore_openai_baseline`,
added after both runs, zero extra API cost): the first run's gpt-5.4-mini
answers were still sitting in the (append-only, PII-redacted) request log,
so re-scoring those 76 eligible answers with the exact same RAGAS judge used
above gives a real apples-to-apples read instead of two differently-defined
metrics. Context Precision/Recall land within
~1 point of each other (57.0%/89.0% vs 56.7%/87.7%) — confirms these two
are a property of retrieval, independent of the generator. Faithfulness,
Answer Relevancy, and Answer Accuracy all show large gaps favoring
gpt-5.4-mini (91.7% vs 72.0%, 59.9% vs 42.3%, 38.2% vs 25.0%) — confirms the
citation-hallucination finding above isn't a judge artifact. One more
finding worth flagging on its own: even gpt-5.4-mini's Answer Accuracy
(38.2%) misses require.md's ≥80% bar by a wide margin under this judge —
since that shows up on both models, it reads as this judge/metric
combination being strict rather than a claim about model quality.

**Temperature sensitivity** (`uv run python -m src.generation.eval_temperature`
— require.md's cost-sensitivity constraint names top_k, reranker, and
temperature; the first two were covered in step 4, this closes the third).
Verified the parameter actually takes effect through Ollama's OpenAI-compat
endpoint before spending compute on it (unlike `num_ctx`/`think`, which that
same endpoint silently ignores — see `providers.py`). On a stratified
15-item subset across temperature 0.0/0.7/1.3: Context Precision/Recall are
identical at all three (a sanity check — retrieval has no dependency on
generation temperature), Faithfulness and Answer Relevancy both peak at 0.7
rather than falling monotonically, and Citation Validity drops sharply at
1.3 (53.8% → 30.8%) — higher sampling randomness makes verbatim-copying a
`chunk_id` string harder, worsening the same weakness already identified
above. Production doesn't set an explicit temperature and runs at Ollama's
default (0.8), close to the 0.7 tested here — nothing in this sweep
motivates changing it. Full table: **[Generation Evaluation, §09](https://claude.ai/code/artifact/17dc2bb1-8253-414b-865d-0fc7213996c1)**.

## Step 5b: Query rewriting A/B test

```bash
uv run python -m src.generation.eval_query_rewrite
```

Does condensing multi-turn conversation history into a standalone query
(`src/generation/query_rewrite.py` — deliberately a fast/cheap model since
this sits on the critical path before retrieval even starts; `gpt-5.4-nano`
when this test ran, `llama3.2-16k` now that the rewriter role has moved
local, not re-run against it) beat the naive "join every turn + question"
construction used everywhere else in this project? Reuses the exact
retrieval harness from steps 2-4 (`retrieval_eval_per_item` now takes an
optional `query_fn`) — same retriever, same metrics, only the query text
differs.

**Result: real gain at k=5, a wash at k=10, real latency cost.** Over the 73
answerable questions that actually have conversation history (rewriting is
a no-op without it, confirmed identical on the other 12): recall@5 83.6% →
90.4%, MRR 0.685 → 0.748. At k=10 — our current config — recall barely
moves (95.9% → 94.5%, within noise): most correct chunks were already
captured by a 10-wide window either way, so the ceiling effect erases most
of the recall benefit. Cost is negligible (~$0.08/1,000 calls at nano
pricing), but latency is real: a sequential call before retrieval adds a
median of 1.1s and a p90 of 4.6s to the critical path, a meaningful bite out
of the 10s budget stacked on top of the existing 3.1s p90.

**Not clearly worth adding at top_k=10** — the recall gain evaporates at
that width and the latency cost is real for a benefit that's mostly about
ranking order (MRR) rather than recall. Revisit if top_k is ever reduced
toward 5, where the recall gain is substantial. Full comparison table and a
sample rewrite folded into the same report:
**[Generation Evaluation, §07](https://claude.ai/code/artifact/17dc2bb1-8253-414b-865d-0fc7213996c1)**.

---

All published report artifacts are also saved locally under
[`reports/`](reports/) for offline viewing — they don't depend on the
Artifact links staying alive.
