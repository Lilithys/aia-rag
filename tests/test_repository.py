import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.evaluation.analyze_final import ledger_summary
from src.service.http_api import SessionRegistry
from src.verify import verify_repository


class RepositoryTests(unittest.TestCase):
    def test_bundled_corpus_matches_frozen_artifacts_and_execution_code(self):
        result = verify_repository()
        self.assertEqual(result["documents"], 488)
        self.assertEqual(result["embedding_shape"], [1591, 1024])
        self.assertEqual(result["splits"], {"dev": 100, "holdout": 100, "smoke": 20})

    def test_missing_usage_ledger_is_an_error_instead_of_free_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "cannot interpret"):
                ledger_summary(Path(folder) / "missing.jsonl", 2)

    def test_invalid_session_identifier_is_rejected_before_runtime(self):
        runtime = Mock()
        registry = SessionRegistry(runtime)
        for session_id in ([], {}, 1, True, ""):
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                registry.ask("hello", session_id)
        runtime.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
