"""Fetch the exact local model files; does not call the generation provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.paths import FINAL_INDEX
from src.retrieval.candidates_v1 import RERANK_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Check the existing cache only")
    args = parser.parse_args()
    encoder = json.loads((FINAL_INDEX / "embeddings/bge_m3/embedding_manifest.json").read_text())["encoder"]
    common = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]
    models = [
        (encoder["model_id"], encoder["revision"], common + ["pytorch_model.bin", "modules.json", "1_Pooling/config.json",
                                                          "config_sentence_transformers.json", "sentence_bert_config.json"]),
        (RERANK_CONFIG["model"], RERANK_CONFIG["revision"], common + ["model.safetensors"]),
    ]
    for model, revision, names in models:
        paths = [Path(hf_hub_download(model, filename=name, revision=revision, local_files_only=args.offline)) for name in names]
        print(json.dumps({"model": model, "revision": revision, "files": len(paths),
                          "bytes": sum(p.stat().st_size for p in paths), "offline": args.offline}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
