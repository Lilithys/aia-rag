import dataclasses
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from src.chunking.strategies.base import ChunkRecord
from src.embedding.models.base import EmbeddingModel
from src.embedding.embed_corpus import embed_chunks_cached, load_verified_embeddings
from src.evaluation.evidence import attach_provenance, answer_targets, canonical, score_window
from src.retrieval.chroma_store import get_collection


def chunk(text, cid="c1", doc="doc"):
    return ChunkRecord(cid, doc, "ssa", "en", "doc.md", "fixed_size", 512, 0, [], text, len(text))


class FakeEncoder(EmbeddingModel):
    name = "fake"
    dim = 2
    is_local = True

    def __init__(self):
        self.calls = 0
        self.revision = "a"

    def cache_identity(self):
        return {**super().cache_identity(), "revision": self.revision}

    def encode_passages(self, texts, batch_size=16):
        self.calls += 1
        return np.array([[len(t), sum(map(ord, t))] for t in texts], dtype=np.float32)

    encode_queries = encode_passages


class ArtifactCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = FakeEncoder()

    def test_same_ids_changed_content_reencodes(self):
        old = embed_chunks_cached(self.model, [chunk("fee 200")], str(self.root))
        same = embed_chunks_cached(self.model, [chunk("fee 200")], str(self.root))
        np.testing.assert_array_equal(old, same)
        self.assertEqual(self.model.calls, 1)
        new = embed_chunks_cached(self.model, [chunk("fee 500")], str(self.root))
        self.assertEqual(self.model.calls, 2)
        self.assertFalse(np.array_equal(old, new))

    def test_model_revision_change_reencodes(self):
        embed_chunks_cached(self.model, [chunk("text")], str(self.root))
        self.model.revision = "b"
        embed_chunks_cached(self.model, [chunk("text")], str(self.root))
        self.assertEqual(self.model.calls, 2)

    def test_unchanged_text_can_be_reused_after_chunk_ids_change(self):
        embed_chunks_cached(self.model, [chunk("same text", "old")], str(self.root))
        embed_chunks_cached(self.model, [chunk("same text", "new")], str(self.root))
        self.assertEqual(self.model.calls, 1)
        manifest = json.loads((self.root / "fake/embedding_manifest.json").read_text())
        self.assertEqual(manifest["reused_rows"], 1)
        self.assertEqual(manifest["encoded_rows"], 0)

    def test_old_ids_only_cache_is_not_certified(self):
        folder = self.root / "fake"
        folder.mkdir()
        np.save(folder / "chunk_embeddings.npy", np.zeros((1, 2), dtype=np.float32))
        (folder / "chunk_ids.json").write_text('["c1"]')
        embed_chunks_cached(self.model, [chunk("text")], str(self.root))
        self.assertEqual(self.model.calls, 1)
        self.assertTrue((folder / "embedding_manifest.json").exists())

    def test_corrupted_vectors_are_rejected(self):
        chunks = [chunk("text")]
        embed_chunks_cached(self.model, chunks, str(self.root))
        path = self.root / "fake/chunk_embeddings.npy"
        np.save(path, np.zeros((1, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_verified_embeddings(chunks, path)

    def test_changed_order_is_rejected(self):
        chunks = [chunk("one", "c1"), chunk("two", "c2")]
        embed_chunks_cached(self.model, chunks, str(self.root))
        with self.assertRaisesRegex(ValueError, "ordered chunk content"):
            load_verified_embeddings(list(reversed(chunks)), self.root / "fake/chunk_embeddings.npy")

    def test_same_id_chroma_content_and_vectors_refresh(self):
        original = get_collection([chunk("old")], np.array([[1, 0]], dtype=np.float32), persist_dir=str(self.root))
        changed = get_collection([chunk("new")], np.array([[0, 1]], dtype=np.float32), persist_dir=str(self.root))
        stored = changed.get(include=["documents", "embeddings"])
        self.assertEqual(stored["documents"], ["new"])
        np.testing.assert_allclose(stored["embeddings"], [[0, 1]])
        # Shape changes require a new derived collection, not just deleting rows.
        resized = get_collection([chunk("new")], np.array([[0, 0, 1]], dtype=np.float32), persist_dir=str(self.root))
        np.testing.assert_allclose(resized.get(include=["embeddings"])["embeddings"], [[0, 0, 1]])

    def test_default_retriever_migrates_old_local_cache_once(self):
        from src.retrieval.retriever import get_default_retriever
        chunks = [chunk("current body")]
        with patch("src.retrieval.retriever.load_chunks", return_value=chunks), \
             patch("src.retrieval.retriever.BgeM3Model", return_value=self.model), \
             patch("src.retrieval.retriever.get_collection", return_value=object()):
            path = str(self.root / "fake/chunk_embeddings.npy")
            get_default_retriever("unused.jsonl", path, str(self.root / "chroma"))
            get_default_retriever("unused.jsonl", path, str(self.root / "chroma"))
        self.assertEqual(self.model.calls, 1)


class EvidenceTests(unittest.TestCase):
    def test_rendered_list_bullets_do_not_hide_evidence(self):
        source = "First condition applies. Second condition also applies."
        text = "- First condition applies.\n- Second condition also applies."
        chunks = [chunk(text)]
        attach_provenance(chunks, text, [{"source_block_id": "b1", "text": source}])
        self.assertEqual(score_window(chunks, {("doc", "b1")})["all_annotated_answer_blocks_hit"], 1)
        self.assertNotEqual(canonical("June - July"), canonical("June July"))
        self.assertNotEqual(canonical("- 200 dollars"), canonical("200 dollars"))

    def test_wrong_amount_and_negation_are_not_fuzzy_hits(self):
        for reference, observed in [("The fee is 200 dollars.", "The fee is 500 dollars."),
                                    ("You are not eligible for this benefit.", "You are eligible for this benefit.")]:
            with self.subTest(reference=reference):
                chunks = [chunk(observed)]
                attach_provenance(chunks, observed, [{"source_block_id": "b1", "text": reference}])
                self.assertEqual(score_window(chunks, {("doc", "b1")})["any_answer_evidence_hit"], 0)

    def test_split_block_requires_union_of_actual_fragments(self):
        text = "First evidence sentence has important information. Second evidence sentence has additional conditions."
        chunks = [chunk(text[:50], "c1"), chunk(text[50:], "c2")]
        attach_provenance(chunks, text, [{"source_block_id": "b1", "text": text}])
        target = {("doc", "b1")}
        self.assertEqual(score_window(chunks[:1], target)["all_annotated_answer_blocks_hit"], 0)
        self.assertEqual(score_window(chunks, target)["all_annotated_answer_blocks_hit"], 1)

    def test_background_hit_cannot_replace_answer_evidence(self):
        item = {"evidence": [{"doc_id": "doc", "spans": [
            {"source_turn": "question", "source_block_id": "background"},
            {"source_turn": "answer", "source_block_id": "answer"}]}]}
        chunks = [chunk("Some background information")]
        attach_provenance(chunks, "Some background information", [{"source_block_id": "background", "text": "Some background information"}])
        self.assertEqual(score_window(chunks, answer_targets(item))["any_answer_evidence_hit"], 0)

    def test_duplicate_chunks_cannot_fill_missing_evidence(self):
        text = "First evidence sentence has important information. Second evidence sentence has additional conditions."
        first = chunk(text[:50])
        attach_provenance([first], text, [{"source_block_id": "b1", "text": text}])
        self.assertEqual(score_window([first, first], {("doc", "b1")})["all_annotated_answer_blocks_hit"], 0)

    def test_empty_retrieval_keeps_zero_score(self):
        result = score_window([], {("doc", "b1")})
        self.assertEqual(result["any_answer_evidence_hit"], 0)
        self.assertEqual(result["annotated_evidence_precision"], 0)


if __name__ == "__main__":
    unittest.main()
