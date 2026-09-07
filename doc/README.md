# Documentation

- [Design note](DESIGN_NOTE.md): architecture and measured choices, within the required 200–500 words.
- [Evaluation summary](EVALUATION_SUMMARY.md): final results, costs, sensitivities, failures and limitations.
- [Validation](VALIDATION.md): corpus/code integrity, tests and reconstruction evidence.
- [Demo script](DEMO_SCRIPT.md): a seven-minute walkthrough.
- [Redacted sample logs](sample_logs.jsonl): five examples of the real request contract and failures.
- [Project history](PROJECT_SUMMARY.md): chronological optimization record, including negative results.
- [Retrieval review](evaluation/v1/retrieval_review.md): quantitative chunking, embedding and retrieval comparison retained because the frozen retrieval code cites it directly.
- [Frozen evaluation release](evaluation/v1/releases/v1.1/release_manifest.json): exact 100-dev, 100-holdout and 20-smoke inputs used by the final pipeline. The v1.0 parent is retained because code validates the two v1.1 revisions against it.

The full source corpus is now in [data/corpus](../data/corpus/README.md), with the directory contract in [data/README](../data/README.md). This directory no longer contains the old 91-document subset. Historical reports carry a notice that retired commands are not current run instructions.
