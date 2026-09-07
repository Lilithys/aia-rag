import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.service.runtime import RagRuntime, ServiceConfig, Session
from src.service.final_evaluate import export, load_record, record_path
from src.service.final_runtime import make_runtime
from src.artifacts import fingerprint
from src.evaluation.paid_calls_v1 import atomic_json


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "requests.jsonl"
        self.retriever = Mock()
        self.chunk = SimpleNamespace(chunk_id="c1", text="You may appeal by phone.", doc_id="doc", heading_path=["Appeals"], source_file="documents/a.md")
        self.retriever.retrieve.return_value = ([self.chunk], {"top1_dense_score": 0.8, "query": "sensitive raw history"})
        self.caller = Mock()
        self.caller.call.return_value = {"status": "ok", "parsed": {"answer": "You may appeal by phone.", "action": "answer", "citations": ["c1"]}, "attempts": [], "cache_hit": False}
        self.runtime = RagRuntime(self.retriever, self.caller, log_path=self.log)

    def test_multi_turn_reset_and_sessions_are_isolated(self):
        first, second = Session(), Session()
        first.ask(self.runtime, "How can I appeal?")
        first.ask(self.runtime, "Can I do that by phone?")
        item = self.retriever.retrieve.call_args.args[0]
        self.assertEqual(len(item["conversation_history"]), 2)
        self.assertEqual(second.history, [])
        previous = first.session_id
        first.reset()
        self.assertNotEqual(first.session_id, previous)
        self.assertEqual(first.history, [])

    def test_answer_citations_are_mandatory_and_never_silently_filtered(self):
        for citations in ([], ["invented"], ["c1", "invented"]):
            with self.subTest(citations=citations):
                self.caller.call.return_value["parsed"]["citations"] = citations
                result = Session().ask(self.runtime, "How can I appeal?")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["raw_citations"], citations)
                self.assertIsNone(result["response"])
        self.assertEqual(len(self.log.read_text().splitlines()), 3)

    def test_pure_clarification_can_have_no_citations(self):
        self.caller.call.return_value["parsed"] = {"answer": "Which two things do you mean?", "action": "clarify", "citations": []}
        result = Session().ask(self.runtime, "Neither of those.")
        self.assertEqual(result["status"], "ok")

    def test_no_results_and_low_confidence_are_logged_without_api(self):
        self.retriever.retrieve.return_value = ([], {"top1_dense_score": 0})
        result = Session().ask(self.runtime, "请问今天的天气？")
        self.assertEqual(result["refusal_reason"], "no_retrieval_results")
        self.assertIn("知识库", result["response"]["answer"])
        self.caller.call.assert_not_called()
        self.retriever.retrieve.return_value = ([self.chunk], {"top1_dense_score": 0.1})
        result = Session().ask(self.runtime, "Weather?")
        self.assertEqual(result["contexts"], [])
        self.assertEqual(len(result["retrieval_contexts"]), 1)
        self.assertEqual(result["refusal_reason"], "low_retrieval_confidence")

    def test_provider_failure_keeps_retrieval_and_not_history(self):
        self.caller.call.return_value = {"status": "failed", "parsed": None, "attempts": [], "cache_hit": False}
        session = Session()
        result = session.ask(self.runtime, "How can I appeal?")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["retrieval_contexts"]), 1)
        self.assertEqual(session.history, [])
        self.assertEqual(json.loads(self.log.read_text())["status"], "failed")

    def test_input_budget_rejects_without_truncating_history_or_calling_model(self):
        runtime = RagRuntime(self.retriever, self.caller, ServiceConfig(max_history_messages=1), self.log)
        history = [{"role": "user", "utterance": "old"}, {"role": "agent", "utterance": "answer"}]
        result = runtime.answer("new", history, "session", 2)
        self.assertEqual(result["status"], "failed")
        self.retriever.retrieve.assert_not_called()
        self.caller.call.assert_not_called()

    def test_trace_redacted_and_request_parameters_explicit(self):
        result = Session().ask(self.runtime, "手机号13800138000，我该怎么申诉？")
        event = json.loads(self.log.read_text())
        self.assertNotIn("13800138000", self.log.read_text())
        self.assertNotIn("sensitive raw history", self.log.read_text())
        self.assertEqual(event["request_id"], result["request_id"])
        self.assertEqual(event["config_sha256"], result["config_sha256"])
        config = self.caller.call.call_args.args[2]
        self.assertEqual(config.thinking, "disabled")
        self.assertEqual(config.temperature, 0)
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(result["citations"][0]["source_file"], "a.md")


class SharedEvaluationTests(unittest.TestCase):
    def test_runtime_factory_persists_manifest_for_log_trace(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            with patch("src.service.final_runtime.preflight", return_value={"corpus_manifest_sha256": "test-corpus"}), \
                 patch("src.service.final_runtime.JsonCaller"), patch("src.service.final_runtime.FolderRetriever"):
                runtime, ledger = make_runtime(state, state, ServiceConfig(), 2)
            manifest = json.loads((state / "runs" / (runtime.run_id + ".json")).read_text())
            self.assertEqual(manifest["config_sha256"], runtime.config_sha256)
            self.assertEqual(manifest["binding"]["artifacts"]["corpus_manifest_sha256"], "test-corpus")
            self.assertEqual(ledger.spent_upper(), 0)

    def test_final_checkpoints_preserve_failures_and_reject_changed_inputs(self):
        items = [{"question_id": "q1", "question": "Current question", "conversation_history": [],
                  "source": {"dialogue_id": "dialogue", "question_turn_id": 1}, "gold_answer": "GOLD_MUST_NOT_LEAK"}]
        plan = {"binding_sha256": "binding", "binding": {"runtime": {"config": {"candidate": "baseline_dense_k10"}},
                 "split": "smoke", "release_manifest_sha256": "release"}}
        result = {"status": "failed", "response": None, "contexts": [], "retrieval_contexts": [],
                                       "latency_ms": {"total": 5}, "generation_cache_hit": False}
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            item = items[0]
            value = {"binding_sha256": "binding", "question_id": "q1",
                     "item_sha256": fingerprint(item), "result": result}
            atomic_json(record_path(out, item), {"payload": value, "sha256": fingerprint(value)})
            self.assertEqual(load_record(out, item, plan), result)
            self.assertEqual(export(out, items, plan), {"planned": 1, "submitted": 1, "successful": 0, "failed": 1})
            with self.assertRaisesRegex(ValueError, "input mismatch"):
                load_record(out, {**item, "question": "changed"}, plan)

    def test_unsubmitted_rows_remain_in_progress_bundle(self):
        items = [{"question_id": "q1"}, {"question_id": "q2"}]
        plan = {"binding_sha256": "binding", "binding": {"runtime": {"config": {"candidate": "baseline_dense_k10"}},
                 "split": "smoke", "release_manifest_sha256": "release"}}
        with tempfile.TemporaryDirectory() as root:
            result = export(Path(root), items, plan)
            self.assertEqual(result["planned"], 2)
            self.assertEqual(result["submitted"], 0)
            self.assertEqual(len(json.loads((Path(root) / "generation_bundle.json").read_text())["cases"]), 2)


if __name__ == "__main__":
    unittest.main()
