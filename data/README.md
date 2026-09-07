# Data included in the repository

`corpus/` is a byte-for-byte copy of the full converted
`multidoc2dial_full_multiformat_multilingual` dataset used for this case study.
It contains all 488 documents, four manifests, the original dataset README,
661 complete dialogues and 4,335 candidate evaluation turns. The copy was
verified against the original directory and the frozen source inventory.

Only `corpus/documents/` is ingested. The manifest supplies document identity,
format and language. Source-block text and test annotations are attached only
as evaluation provenance after retrieval text is fixed; they are not embedding
inputs. The service never reads the full candidate test set.

The actual evaluation release remains at
`../doc/evaluation/v1/releases/v1.1/`: **100 dev + 100 holdout**. Its 20 smoke
questions are an exact dev subset. The v1.0 parent is retained because v1.1
validates its two disclosed annotation corrections against that parent.

| Path | Purpose | Included in Git |
|---|---|---|
| `corpus/documents/` | 488 mixed-format source documents | Yes |
| `corpus/manifests/`, `corpus/test/` | Original transformed dataset and source lineage | Yes; never retrieved |
| `experiments/eval_v1/` | Initial 488-document baseline, frozen cleaned text and early smoke evidence | Yes; baseline/reconstruction evidence |
| `experiments/eval_v11/` | Evaluation release v1.1 development work: source audit, six-arm sensitivity study, full dev run and judge diagnostics | Yes; development evidence, not final holdout |
| `experiments/eval_v12/` | Heading-preservation fix, final index, final smoke, final holdout and service acceptance | Yes; contains final runtime/evaluation evidence |
| `experiments/tokenizer_cache/` | Public cl100k_base vocabulary for offline checks | Yes |
| `local/`, `logs/`, any `chroma/` or `api_cache/` | Scratch builds and newly generated runtime state | No |

The experiment folder names describe the chronological evaluation lineage.
`eval_v11` means evaluation release **v1.1** (the dot is omitted in the folder
name); it is not “version eleven.” `eval_v12` is the subsequent final-candidate
line after the chunk-preservation repair. These names are unrelated to the
number of documents, test questions, or model version. See the
[experiment inventory](experiments/README.md) for the status of every retained
subdirectory and the meaning of its common files.

File hashes in historical reports describe the run when it happened. Their
inventories may mention deleted intermediate caches or retired source files.
Current reproduction consists of corpus/release verification, exact chunk
rebuilding from frozen clean text, the baseline/final dev retrieval comparison,
and offline final-result aggregation. The complete pre-cleanup development
workspace was separately backed up by the project owner.

The dataset README in [corpus/README.md](corpus/README.md) describes the
deterministic format/language transformation of MultiDoc2Dial. This is an
adapted case-study evaluation, not an official MultiDoc2Dial benchmark.
