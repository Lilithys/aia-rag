# MultiDoc2Dial Full Multiformat Multilingual Benchmark

This is a deterministic transformation of the complete **MultiDoc2Dial test split** for RAG evaluation. It retains all 488 documents, all 661 test dialogues, and all 8,938 original turns. No dialogue is sampled out, and no synthetic multi-document question or explicit negative label is added.

## Layout

- `documents/`: the only directory that should be ingested into a retriever.
- `test/dialogues.json`: 661 complete dialogues with every turn translated as one consistent dialogue language.
- `test/eval_turns.json`: all 4,335 adjacent user-to-agent pairs, preserving the complete prior history.
- `manifests/document_manifest.json`: document identity, format, language, source blocks, and format-specific locations.
- `manifests/build_report.json`: distributions and integrity counts.

## Document formats

- Markdown: 366
- text PDF: 49
- scanned/image-only PDF: 24
- DOCX: 25
- TXT: 24

Markdown files intentionally have no metadata front matter. Format-neutral metadata lives in the manifest, so Markdown does not receive metadata that the other formats lack.

## Independent language assignment

Documents: English 244, Chinese 146, mixed Chinese-English 98.

Dialogues: English 331, Chinese 198, mixed Chinese-English 132.

Document language, document format, and dialogue language are assigned independently with deterministic domain-stratified hashing. They are not deliberately aligned. Every turn in one dialogue—including both user and agent turns—uses the dialogue's assigned language. Original English is retained in `source_utterance_en` and `source_text_en`.

Translations were generated with `deepseek-v4-flash` through the `deepseek` provider. Thinking/reasoning mode was disabled for deterministic translation, and every batch was checked for valid JSON, output count, and non-empty content.

## Evaluation categories

- answer: 3214
- clarification: 880
- unanswerable: 241

These categories are derived from the original agent dialogue act. The original `respond_no_solution` examples are preserved as unanswerable cases; no new unanswerable questions are manufactured.

## IDs and retrieval chunking

`source_block_id` identifies a stable pre-chunking source block and connects test references to the correct block after translation and format conversion. It is **not** a retrieval chunk ID. A downstream ingestion pipeline should create its own `retrieval_chunk_id` values and retain `doc_id` plus `source_block_id` as lineage metadata.

## Scope

Only the original test split is transformed. The original train and validation splits remain unchanged because this package is an evaluation corpus, not a training package.
