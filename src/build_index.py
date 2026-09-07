"""Rebuild chunks from checked-in clean text or reconvert the bundled originals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from src.artifacts import file_sha256
from src.paths import CORPUS, FINAL_INDEX, PREPARED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=Path("data/local/rebuild"))
    parser.add_argument("--reconvert", action="store_true", help="Parse/OCR all source documents again before chunking")
    parser.add_argument("--encode", action="store_true", help="Encode changed chunks locally; identical frozen chunks reuse verified vectors")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new --out directory; existing artifacts are never overwritten")
    args.out.mkdir(parents=True)
    prepared = PREPARED
    if args.reconvert:
        from src.preprocess.run import main as convert
        from src.chunking.prepare import main as clean
        prepared = args.out / "inputs"
        parsed = prepared / "processed/markdown"
        cleaned = prepared / "interim/markdown_clean"
        if convert(["--doc-root", str(args.source_root), "--out-dir", str(parsed)]) != 0:
            raise ValueError("Source conversion incomplete; inspect preprocessing_report.json")
        if clean(["--in-dir", str(parsed), "--out-dir", str(cleaned)]) != 0:
            raise ValueError("Cleaning failed")
        manifest = {"source_manifest_sha256": file_sha256(args.source_root / "manifests/document_manifest.json"),
                    "clean_files": [{"name": p.name, "sha256": file_sha256(p)} for p in sorted(cleaned.glob("*.md"))]}
        (prepared / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from src.evaluation.prepare_delivery_index import main as chunk
    folder = args.out / "semantic_1024_v2"
    chunk(["--source-root", str(args.source_root), "--source-artifacts", str(prepared), "--out", str(folder)])
    same = file_sha256(folder / "chunks.jsonl") == file_sha256(FINAL_INDEX / "chunks.jsonl")
    if same:
        shutil.copytree(FINAL_INDEX / "embeddings", folder / "embeddings")
    elif args.encode:
        import torch
        from src.embedding.embed_corpus import embed_chunks_cached, load_chunks
        from src.embedding.models.bge_m3 import BgeM3Model
        torch.set_num_threads(4)
        revision = json.loads((FINAL_INDEX / "embeddings/bge_m3/embedding_manifest.json").read_text())["encoder"]["revision"]
        embed_chunks_cached(BgeM3Model(device="cpu", revision=revision), load_chunks(str(folder / "chunks.jsonl")), str(folder / "embeddings"))
    if (folder / "embeddings").exists():
        from src.embedding.embed_corpus import load_chunks, load_verified_embeddings
        load_verified_embeddings(load_chunks(str(folder / "chunks.jsonl")), folder / "embeddings/bge_m3/chunk_embeddings.npy")
    print(json.dumps({"artifact_folder": str(folder), "matches_frozen_chunks": same,
                      "embeddings_ready": (folder / "embeddings").exists(), "paid_api_calls": 0,
                      "note": "Fresh OCR can differ by Tesseract/library version. Changed chunks require --encode and a new evaluation."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
