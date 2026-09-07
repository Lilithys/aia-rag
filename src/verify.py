"""Verify the checked-in corpus, release, frozen execution code and result inputs."""
from __future__ import annotations

import json
from pathlib import Path

from src.artifacts import file_sha256
from src.data_prep.revise_eval_v11 import validate_revision
from src.data_prep.validate_eval_v1 import validate_release
from src.paths import CORPUS, FINAL_INDEX, HOLDOUT, PARENT_RELEASE, RELEASE


def verify_repository():
    source = validate_release(PARENT_RELEASE, CORPUS)
    validate_revision(PARENT_RELEASE, RELEASE)
    plan = json.loads((HOLDOUT / "plan.json").read_text())
    runtime = plan["binding"]["runtime"]
    for filename, key in (("chunks.jsonl", "chunks_sha256"),
                          ("embeddings/bge_m3/embedding_manifest.json", "embedding_manifest_sha256"),
                          ("embeddings/bge_m3/chunk_embeddings.npy", "embedding_sha256")):
        if file_sha256(FINAL_INDEX / filename) != runtime[key]:
            raise ValueError(f"Final artifact differs from the measured holdout: {filename}")
    from src.embedding.embed_corpus import load_chunks, load_verified_embeddings
    chunks = load_chunks(str(FINAL_INDEX / "chunks.jsonl"))
    values = load_verified_embeddings(chunks, FINAL_INDEX / "embeddings/bge_m3/chunk_embeddings.npy")
    document_ids = {d["doc_id"] for d in json.loads((CORPUS / "manifests/document_manifest.json").read_text())["documents"]}
    if {c.doc_id for c in chunks} != document_ids or len(chunks) != runtime["chunk_count"]:
        raise ValueError("Final chunks do not cover the exact source document identities")
    for chunk in chunks:
        source_path = (CORPUS / chunk.source_file).resolve()
        if not source_path.is_relative_to((CORPUS / "documents").resolve()) or not source_path.is_file():
            raise ValueError(f"Missing/non-document citation source: {chunk.source_file}")
    code = dict(runtime["code_sha256"])
    code.update(plan["binding"]["code_sha256"])
    for metric in ("correctness", "faithfulness", "context_precision"):
        code.update(json.loads((HOLDOUT / metric / "plan.json").read_text())["binding"]["code_sha256"])
    checked = set()
    for original, expected in code.items():
        # Old execution records retain their original absolute path as provenance.
        # Verification resolves that same repository-relative source in this clone.
        relative = "src/" + original.split("src/", 1)[1]
        if file_sha256(relative) != expected:
            raise ValueError(f"Measured execution code changed: {relative}")
        checked.add(relative)
    prepared = Path("data/experiments/eval_v1/retrieval_fixed")
    lineage = json.loads((prepared / "corpus_manifest.json").read_text())
    for document in lineage["documents"]:
        if file_sha256(prepared / document["clean_file"]) != document["clean_sha256"]:
            raise ValueError(f"Prepared clean text changed: {document['doc_id']}")
    return {"documents": source["documents"], "source_bytes_verified": True,
            "splits": {s: len(json.loads((RELEASE / f"{s}.json").read_text())["items"]) for s in ("dev", "holdout", "smoke")},
            "chunks": len(chunks), "embedding_shape": list(values.shape),
            "measured_execution_files_unchanged": len(checked), "prepared_documents_verified": len(lineage["documents"]),
            "paid_api_calls": 0}


def main():
    print(json.dumps(verify_repository(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
