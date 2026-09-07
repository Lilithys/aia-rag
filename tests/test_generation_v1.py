import dataclasses
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.chunking.strategies.base import ChunkRecord
from src.evaluation.paid_calls_v1 import (BudgetLedger, BudgetExceeded, CallConfig, JsonCaller,
                                         extract_usage, input_upper_bound, interval_multiplier, price)
from src.evaluation.prompts_v1 import (generation_messages, judge_messages, validate_generation,
                                      validate_correctness, validate_faithfulness)
from src.evaluation.prepare_calibration_v11 import read_record
from src.evaluation.score_v11 import summarize
from src.evaluation.paid_calls_v1 import atomic_json
from src.artifacts import fingerprint


def response(content, prompt=100, completion=20, cached=30):
    return SimpleNamespace(id="response-id", model="deepseek-v4-flash", system_fingerprint=None,
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": prompt, "completion_tokens": completion,
                                                "prompt_cache_hit_tokens": cached}),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])


GOOD = {"correct": True, "observed_action": "answer", "reason": "The evidence supports the answer."}


class CallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = BudgetLedger(self.root / "usage.jsonl", 5)
        self.client = Mock()
        self.client.chat.completions.create.return_value = response(json.dumps(GOOD))
        self.caller = JsonCaller(self.root / "cache", self.ledger, client=self.client)
        self.messages = [{"role": "user", "content": "Judge this JSON response"}]

    def call(self, **kwargs):
        return self.caller.call("correctness", kwargs.get("messages", self.messages),
                                kwargs.get("config", CallConfig()), validate_correctness)

    def test_explicit_non_thinking_usage_and_exact_cache(self):
        first = self.call(); second = self.call()
        self.assertEqual(first["status"], "ok")
        self.assertTrue(second["cache_hit"])
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        sent = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(sent["max_tokens"], 2048)
        self.assertEqual(len(self.ledger.receipts()), 1)
        self.assertAlmostEqual(self.ledger.spent_upper(), self.ledger.receipts()[0]["usage_cost_estimate_rmb"])
        self.assertLessEqual(self.ledger.spent_upper(), price("deepseek-v4-flash", 100, 20, 30))

    def test_cache_invalidates_on_context_order_or_model(self):
        self.call()
        self.call(messages=self.messages + [{"role": "user", "content": "different context"}])
        self.call(config=CallConfig(model="deepseek-v4-pro"))
        self.assertEqual(self.client.chat.completions.create.call_count, 3)

    def test_empty_json_retry_keeps_both_usage_records(self):
        self.client.chat.completions.create.side_effect = [response(""), response(json.dumps(GOOD))]
        with patch("src.evaluation.paid_calls_v1.time.sleep"):
            result = self.call()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(len(self.ledger.receipts()), 2)

    def test_failed_json_not_cached(self):
        self.client.chat.completions.create.return_value = response('{"correct": "yes"}')
        with patch("src.evaluation.paid_calls_v1.time.sleep"):
            result = self.call()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(list((self.root / "cache").glob("*.json")))

    def test_corrupt_cache_is_rejected_without_paid_retry(self):
        self.call()
        cache = next((self.root / "cache").glob("*.json"))
        value = json.loads(cache.read_text()); value["payload"]["result"]["parsed"]["correct"] = False
        cache.write_text(json.dumps(value))
        with self.assertRaises(ValueError): self.call()
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_budget_reserved_before_network_and_survives_restart(self):
        small = BudgetLedger(self.root / "small.jsonl", .001)
        caller = JsonCaller(self.root / "cache", small, client=self.client)
        result = caller.call("judge", self.messages, CallConfig(), validate_correctness)
        self.assertEqual(result["status"], "budget_exceeded")
        self.client.chat.completions.create.assert_not_called()
        attempt, amount = self.ledger.reserve("judge", CallConfig(), "hash", 1000)
        recovered = BudgetLedger(self.ledger.path, 5)
        self.assertEqual(recovered.spent_upper(), amount)

    def test_timeout_unknown_usage_does_not_become_free(self):
        self.client.chat.completions.create.side_effect = TimeoutError()
        result = self.call(config=CallConfig(max_attempts=1))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(self.ledger.receipts()[0]["usage_unknown"])
        self.assertGreater(self.ledger.spent_upper(), 0)

    def test_input_cap_makes_no_request(self):
        result = self.call(config=CallConfig(input_token_upper_limit=10))
        self.assertEqual(result["status"], "input_budget_exceeded")
        self.client.chat.completions.create.assert_not_called()

    def test_price_schedule_and_usage_cache_fields(self):
        sunday = dt.datetime(2026, 9, 6, 2, tzinfo=dt.timezone.utc).timestamp()
        monday = dt.datetime(2026, 9, 7, 2, tzinfo=dt.timezone.utc).timestamp()
        self.assertEqual(interval_multiplier(sunday, sunday+10), .5)
        self.assertEqual(interval_multiplier(monday, monday+10), 1)
        self.assertEqual(extract_usage(response("{}"))["cached_tokens"], 30)


class EvaluationTests(unittest.TestCase):
    def test_no_label_or_reference_leaks_to_generation_or_faithfulness(self):
        item = {"question": "What is available?", "conversation_history": [],
                "task_label": "clarify", "gold_answer": "SECRET GOLD SENTINEL", "evidence": []}
        generated = json.dumps(generation_messages(item, []))
        self.assertNotIn("SECRET GOLD SENTINEL", generated)
        self.assertNotIn("task_label", generated)
        faithful = json.dumps(judge_messages("faithfulness", item, {"answer": "test"}, []))
        self.assertNotIn("SECRET GOLD SENTINEL", faithful)
        self.assertNotIn("task_label", faithful)
        self.assertIn("SECRET GOLD SENTINEL", json.dumps(judge_messages("correctness", item, {"answer": "test"}, [])))

    def test_no_fact_and_unknown_citations_validated(self):
        validate_faithfulness({"claims": []}, set())
        with self.assertRaises(ValueError):
            validate_faithfulness({"claims": [{"claim": "The fee is $200", "supported": True,
                                              "supporting_chunk_ids": ["invented"]}]}, {"real"})
        with self.assertRaises(ValueError):
            validate_generation({"answer": "", "citations": [], "action": "answer"})

    def test_failed_and_unknown_cases_stay_in_denominator(self):
        cases = [{"candidate": "c", "question_id": str(i), "status": status}
                 for i, status in enumerate(("ok", "failed", "ok"))]
        items = {str(i): {"task_label": "clarify"} for i in range(3)}
        scores = {("c", "0", "correctness"): {"status": "ok", "score": 1}}
        result = summarize(cases, items, scores, ["correctness"])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["correctness"]["bounds"], [1/3, 2/3])
        self.assertEqual(result["correctness"]["generation_failures"], 1)

    def test_checkpoint_corruption_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            payload = {"status": "ok"}
            atomic_json(path, {"payload": payload, "sha256": fingerprint(payload)})
            self.assertEqual(read_record(path), {"status": "ok"})
            value = json.loads(path.read_text()); value["payload"]["status"] = "failed"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): read_record(path)


if __name__ == "__main__":
    unittest.main()
