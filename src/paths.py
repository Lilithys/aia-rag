"""Repository-relative inputs shared by public command-line entrypoints."""
from pathlib import Path

CORPUS = Path("data/corpus")
FINAL_INDEX = Path("data/experiments/eval_v12/delivery_candidate/semantic_1024_v2")
BASELINE_INDEX = Path("data/experiments/eval_v1/retrieval_fixed/semantic_1024")
PREPARED = Path("data/experiments/eval_v1/retrieval_fixed")
HOLDOUT = Path("data/experiments/eval_v12/final_holdout_20260907")
PARENT_RELEASE = Path("doc/evaluation/v1/releases/v1.0")
RELEASE = Path("doc/evaluation/v1/releases/v1.1")
