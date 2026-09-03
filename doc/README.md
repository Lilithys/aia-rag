# MultiDoc2Dial Multiformat Multilingual RAG Benchmark v3

This benchmark contains 91 documents and 100 RAG test questions derived from MultiDoc2Dial.

## Document formats

- 68 Markdown documents
- 9 text-layer PDFs
- 5 image-only scanned PDFs
- 5 DOCX documents
- 4 UTF-8 TXT documents

## Languages

Documents: 46 English, 27 Simplified Chinese, and 18 natural Chinese-English mixed documents.

Questions: 50 English, 30 Simplified Chinese, and 20 Chinese-English mixed questions. Gold answers and conversation histories follow the question language. The set includes controlled cross-lingual retrieval cases.

## Retrieval structure

- 75 answerable single-document questions
- 10 answerable two-document composite questions
- 15 source-labeled unanswerable questions
- Six core documents support six questions each
- Forty pure distractor documents support no selected question

Every question includes two hard, two medium, and two easy negative document IDs. Hard negatives are same-domain documents ranked by lexical TF-IDF similarity; medium negatives are same-domain lower-ranked documents; easy negatives are cross-domain documents.

The final set contains 22 answerable cross-lingual cases under a strict question-versus-required-document language check. Eight of the ten multi-document questions use heterogeneous formats, and eight use heterogeneous document languages.

## Evidence

`test/test.json` contains gold answers, required document metadata, and localized evidence mapped to stable `source_block_id` values. These are evidence-alignment blocks, not retrieval chunks. `manifests/document_manifest.json` maps each source block to page, paragraph, line, or Markdown block locations depending on format.

Scanned PDFs intentionally contain no hidden text layer. Their evidence text remains available only in the gold manifest/test metadata for evaluation.

Source-labeled unanswerable cases and synthetic two-document compound questions should receive human review before use as a formal benchmark.

See `VALIDATION.md` for the automated checks, PDF render review, and DOCX QA limitation.
