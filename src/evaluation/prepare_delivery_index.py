"""Build the versioned semantic-section-v2 delivery candidate index inputs."""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

from src.artifacts import file_sha256, fingerprint
from src.chunking.loader import load_clean_corpus
from src.chunking.strategies import semantic_section_v2
from src.data_prep.validate_eval_v1 import validate_release
from .evidence import ALIGNMENT_VERSION, attach_provenance
from .paid_calls_v1 import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/corpus"),
                        help="MultiDoc2Dial full multiformat corpus root")
    parser.add_argument("--source-artifacts", type=Path,
                        default=Path("data/experiments/eval_v1/retrieval_fixed"))
    parser.add_argument("--release", type=Path,
                        default=Path("doc/evaluation/v1/releases/v1.1"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/local/rebuild/semantic_1024_v2"))
    args = parser.parse_args(argv)

    validate_release(Path("doc/evaluation/v1/releases/v1.0"), args.source_root)
    source_manifest_path = args.source_root / "manifests/document_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_docs = {d["doc_id"]: d for d in source_manifest["documents"]}
    docs = load_clean_corpus(str(args.source_artifacts / "interim/markdown_clean"))
    if Counter(d.meta["doc_id"] for d in docs) != Counter(source_docs.keys()):
        raise ValueError("Clean corpus and source manifest document IDs differ")

    chunks, alignments = [], []
    for doc in docs:
        source = source_docs[doc.meta["doc_id"]]
        if doc.meta["source_file"] != source["file"]:
            raise ValueError("Clean document source path differs from manifest")
        records = semantic_section_v2.chunk_document(doc.body, doc.meta, chunk_size=1024, overlap=0)
        alignment = attach_provenance(records, doc.body, source["source_blocks"])
        chunks.extend(records)
        alignments.append({"doc_id": doc.meta["doc_id"], "format": source["format"], **alignment})

    args.out.mkdir(parents=True, exist_ok=True)
    chunks_path = args.out / "chunks.jsonl"
    with chunks_path.open("w") as stream:
        for chunk in chunks:
            stream.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")
    atomic_json(args.out / "alignment_report.json", alignments)
    clean_files = sorted((args.source_artifacts / "interim/markdown_clean").glob("*.md"))
    manifest = {"schema_version": 1, "artifact": "delivery-candidate-semantic-1024-v2",
        "documents": len(docs), "chunks": len(chunks), "strategy": semantic_section_v2.NAME,
        "chunk_size": 1024, "overlap": 0, "alignment_version": ALIGNMENT_VERSION,
        "source_corpus_manifest_sha256": file_sha256(source_manifest_path),
        "frozen_release_manifest_sha256": file_sha256(args.release / "release_manifest.json"),
        "source_artifact_manifest_sha256": file_sha256(args.source_artifacts / "corpus_manifest.json"),
        "clean_corpus_fingerprint": fingerprint([{"name": p.name, "sha256": file_sha256(p)} for p in clean_files]),
        "chunks_sha256": file_sha256(chunks_path),
        "alignment_report_sha256": file_sha256(args.out / "alignment_report.json"),
        "code_sha256": {str(p): file_sha256(p) for p in [Path(__file__),
            Path("src/chunking/strategies/semantic_section_v2.py"), Path("src/chunking/doc_tree.py"),
            Path("src/chunking/splitting.py"), Path("src/chunking/tokenizer.py"),
            Path("src/evaluation/evidence.py")]},
        "index_text_inputs": "Only cleaned corpus text; no questions, labels, references, or source-block annotations",
        "source_spans_scope": "Evaluation-only provenance attached after chunk text was fixed; not encoded by the embedding model",
        "holdout_queried": False}
    manifest["binding_sha256"] = fingerprint(manifest)
    atomic_json(args.out / "index_input_manifest.json", manifest)
    print(json.dumps({"documents": len(docs), "chunks": len(chunks),
                      "source_blocks_aligned": sum(a["aligned"] for a in alignments),
                      "source_blocks_total": sum(a["blocks"] for a in alignments),
                      "binding_sha256": manifest["binding_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
