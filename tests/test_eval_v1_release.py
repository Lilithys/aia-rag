"""Offline regression checks for immutable evaluation publishing and leakage gates."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.data_prep.build_eval_v1_candidates import _convert_candidate, _write_review_csv
from src.data_prep.freeze_eval_v1 import _sha256, _write_json, publish_release
from src.data_prep.validate_eval_v1 import validate_release


class EvaluationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "documents").mkdir(parents=True)
        (self.source / "test").mkdir()
        self.document = self.source / "documents/policy.md"
        self.document.write_text("Policy: eligibility depends on age.\n")
        self.candidates_dir = self.root / "candidates"
        self.candidates_dir.mkdir()
        self.inventory_path = self.root / "inventory.json"
        self.protocol_path = self.root / "protocol.json"
        self.items = {}
        source_items = []
        for split in ("dev", "holdout"):
            candidates = []
            for label in ("answer", "clarify"):
                qid = f"{split}-{label}"
                native = {
                    "eval_item_id": qid, "dialogue_id": qid, "question_turn_id": 1,
                    "answer_turn_id": 2, "agent_dialogue_act": "query_condition" if label == "clarify" else "respond_solution",
                    "evaluation_category": "clarification" if label == "clarify" else "answer",
                    "domain": "ssa", "dialogue_language": "en", "language_relation": "aligned",
                    "question": "What determines my eligibility?", "gold_answer": "Eligibility depends on age.",
                    "conversation_history": [], "required_doc_ids": ["policy"], "gold_doc_ids": ["policy"],
                    "required_documents": [{"doc_id": "policy", "file": "documents/policy.md", "format": "markdown", "language": "en"}],
                    "evidence": [{"doc_id": "policy", "source_turn": "answer", "source_block_id": "policy::b1", "evidence_text": "Eligibility depends on age."}],
                }
                source_items.append(native)
                candidate = _convert_candidate(native, split, len(candidates) + 1)
                candidate["synthetic"] = False
                candidate["review"].update(status="approved", task_label=label, reviewed_reference="=SOURCE", label_reason="Evidence supports the reviewed behavior.")
                candidates.append(candidate)
            synthetic = json.loads(json.dumps(candidates[0]))
            synthetic.update(question_id=f"{split}-refuse", sequence=3, synthetic=True,
                             required_doc_ids=[], gold_doc_ids=[], required_documents=[], evidence=[],
                             question="What is my live claim status?", source_gold_answer="I cannot access live claims.")
            synthetic["source"]["dialogue_id"] = f"synthetic::{split}-refuse"
            synthetic["review"].update(task_label="refuse", evidence_ids=[])
            candidates.append(synthetic)
            self.items[split] = candidates
        self.source_eval = self.source / "test/eval_turns.json"
        _write_json(self.source_eval, {"items": source_items})
        self.inventory = {
            "counts": {"documents": 1}, "corpus_fingerprint_sha256": "fixture-corpus",
            "documents": [{"doc_id": "policy", "file": "documents/policy.md", "sha256": _sha256(self.document)}],
            "source_files": [{"path": "test/eval_turns.json", "sha256": _sha256(self.source_eval)}],
            "holdout_excluded_dialogue_ids": ["dev-answer"],
        }
        _write_json(self.inventory_path, self.inventory)
        targets = {"answer": 1, "clarify": 1, "refuse": 1}
        _write_json(self.protocol_path, {"protocol_version": "1.0", "task_target_per_split": targets, "smoke_task_target": targets})
        self.manifest = {
            "candidate_release": "fixture", "status": "ready_to_freeze", "composer_version": "fixture-v1", "seed": 7,
            "source_eval_sha256": _sha256(self.source_eval), "source_corpus_fingerprint_sha256": "fixture-corpus",
            "source_inventory_sha256": _sha256(self.inventory_path), "task_target_per_split": targets,
            "review_provenance": {"reviewer_type": "ai_agent", "independent_human_review": False},
            "split_policy": {}, "review_ledger": {},
        }
        self.repin_candidates()
        self.args = argparse.Namespace(candidates_dir=self.candidates_dir, inventory=self.inventory_path,
            source_root=self.source, protocol=self.protocol_path, out_dir=self.root / "release", release_version="1.0")

    def repin_candidates(self):
        """Simulate deliberate recomposition; tests then exercise semantic gates."""
        self.manifest["candidate_files"] = {}
        self.manifest["review_files"] = {}
        self.manifest["question_ids"] = {}
        for split, items in self.items.items():
            path = self.candidates_dir / f"{split}_candidates.json"
            _write_json(path, {**self.manifest, "split": split, "items": items})
            review_path = self.candidates_dir / f"{split}_review.csv"
            _write_review_csv(review_path, items)
            self.manifest["candidate_files"][split] = {"path": path.name, "sha256": _sha256(path)}
            self.manifest["review_files"][split] = {"path": review_path.name, "sha256": _sha256(review_path)}
            self.manifest["question_ids"][split] = [i["question_id"] for i in items]
        _write_json(self.candidates_dir / "composition_manifest.json", self.manifest)

    def assert_blocked(self, message):
        with self.assertRaisesRegex(ValueError, message):
            publish_release(self.args)
        self.assertFalse(self.args.out_dir.exists(), "invalid release must publish no files")

    def test_reproducible_release_and_exact_smoke_subset(self):
        first = publish_release(self.args)
        result = validate_release(self.args.out_dir, self.source)
        self.assertTrue(result["source_bytes_verified"])
        self.assertEqual(result["questions"], {"dev": 3, "holdout": 3, "smoke": 3})
        original = self.args.out_dir
        self.args.out_dir = self.root / "second-release"
        self.assertEqual(first, publish_release(self.args))
        for path in original.iterdir():
            self.assertEqual(path.read_bytes(), (self.args.out_dir / path.name).read_bytes())

    def test_candidate_tampering_is_blocked(self):
        path = self.candidates_dir / "dev_candidates.json"
        path.write_text(path.read_text().replace("eligibility", "INJECTED"))
        self.assert_blocked("SHA-256 mismatch")

    def test_original_question_cannot_change_even_with_recomputed_manifest(self):
        self.items["dev"][0]["question"] = "A silently rewritten question"
        self.repin_candidates()
        self.assert_blocked("source content changed")

    def test_changed_document_is_blocked(self):
        self.document.write_text("A materially different policy")
        self.assert_blocked("SHA-256 mismatch")

    def test_pending_review_is_blocked(self):
        self.items["dev"][0]["review"]["status"] = "pending"
        self.repin_candidates()
        self.assert_blocked("Review incomplete")

    def test_duplicate_question_and_dialogue_are_blocked(self):
        for field in ("question_id", "dialogue_id"):
            with self.subTest(field=field):
                item = self.items["dev"][-1]
                target = item if field == "question_id" else item["source"]
                previous = target[field]
                target[field] = "dev-answer"
                self.repin_candidates()
                self.assert_blocked("duplicate")
                target[field] = previous

    def test_cross_split_dialogue_leakage_is_blocked(self):
        self.items["holdout"][-1]["source"]["dialogue_id"] = "synthetic::dev-refuse"
        self.repin_candidates()
        self.assert_blocked("dev/holdout dialogue overlap")

    def test_historically_exposed_holdout_is_blocked(self):
        self.inventory["holdout_excluded_dialogue_ids"].append("holdout-answer")
        _write_json(self.inventory_path, self.inventory)
        self.manifest["source_inventory_sha256"] = _sha256(self.inventory_path)
        self.repin_candidates()
        self.assert_blocked("historically exposed")

    def test_overwrite_is_blocked_and_existing_bytes_preserved(self):
        publish_release(self.args)
        before = (self.args.out_dir / "release_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            publish_release(self.args)
        self.assertEqual(before, (self.args.out_dir / "release_manifest.json").read_bytes())

    def test_release_tampering_is_detected(self):
        publish_release(self.args)
        path = self.args.out_dir / "holdout.json"
        path.write_text(path.read_text().replace("eligibility", "INJECTED"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_release(self.args.out_dir)


if __name__ == "__main__":
    unittest.main()
