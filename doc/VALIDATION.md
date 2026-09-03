# Validation report

Validation date: 2026-09-01

## Result

The final dataset passed the independent validator:

```bash
/private/tmp/rag2-artifact-venv/bin/python validate_multidoc2dial_multiformat_multilingual_v3.py
```

- 91 unique document IDs and files
- 100 unique question IDs
- Documents: 68 Markdown, 9 text PDFs, 5 image-only scanned PDFs, 5 DOCX, 4 TXT
- Document languages: 46 English, 27 Simplified Chinese, 18 Chinese-English mixed
- Question languages: 50 English, 30 Simplified Chinese, 20 Chinese-English mixed
- Question types: 75 single-document, 10 two-document, 15 unanswerable
- Roles: 6 core supporting, 30 supporting, 15 unanswerable candidates, 40 distractors
- 22 answerable cross-lingual cases under the strict check that the question language is absent from all required documents
- 8 of 10 multi-document questions use heterogeneous formats
- 8 of 10 multi-document questions use heterogeneous document languages
- Every question has 2 hard, 2 medium, and 2 easy negatives with the expected domain relationships
- Every required/candidate/evidence/negative document path and `source_block_id` reference resolves
- All evidence text and format-specific locations match the document manifest
- All 9 text PDFs have extractable text; all 5 scanned PDFs have zero extractable characters
- All Markdown heading and list markers pass whitespace syntax checks

## Artifact QA

All 30 pages across the 14 PDFs were rendered with PyMuPDF and visually inspected for clipping, overlap, missing glyphs, and broken page flow. No material layout defect was found.

All 5 DOCX files passed ZIP integrity, `python-docx` parsing, section/margin audit, style lint, and accessibility audit. The environment did not contain LibreOffice/`soffice`, so page-level DOCX rendering and visual inspection could not be completed.

## Review note

Chinese and mixed-language text was machine-translated locally with Qwen 2.5 7B. The synthetic two-document questions and source-labeled unanswerable questions should receive human review before the dataset is treated as a formal benchmark or used for publication-quality results.
