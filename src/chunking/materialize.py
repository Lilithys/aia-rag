"""Step 2c: materialize the chosen chunking config as the corpus that indexing
will consume next.

Default is the config the experiment (see chunking_experiment_results.json /
the published comparison report) recommends: semantic_section, chunk_size
1024 (soft cap), overlap 0.

Usage:
    uv run python -m src.chunking.materialize
    uv run python -m src.chunking.materialize --strategy recursive_structural --chunk-size 1024 --overlap 150
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .loader import load_clean_corpus
from .strategies import fixed_size, recursive_structural, semantic_section

STRATEGIES = {
    "fixed_size": fixed_size,
    "recursive_structural": recursive_structural,
    "semantic_section": semantic_section,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="semantic_section", choices=list(STRATEGIES))
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--out", default="data/processed/chunks/chunks.jsonl")
    args = parser.parse_args(argv)

    mod = STRATEGIES[args.strategy]
    docs = load_clean_corpus()

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n_chunks = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            for chunk in mod.chunk_document(d.body, d.meta, chunk_size=args.chunk_size, overlap=args.overlap):
                f.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {len(docs)} documents to {args.out}", file=sys.stderr)
    print(f"  strategy={args.strategy} chunk_size={args.chunk_size} overlap={args.overlap}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
