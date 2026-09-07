import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.artifacts import file_sha256
from src.evaluation.context_precision_v11 import evaluate, validate_verdict
from src.evaluation.paid_calls_v1 import CallConfig
from src.evaluation.score_v11 import prepare, summarize, load_bundle

PARENT = Path("doc/evaluation/v1/releases/v1.0")
RELEASE = Path("doc/evaluation/v1/releases/v1.1")
RUN = Path("data/experiments/eval_v12/final_smoke_20260907")


class ContextPrecisionTests(unittest.TestCase):
    def test_actual_ragas_aggregation_and_transport_configuration(self):
        caller = Mock()
        caller.call.side_effect = [{"status": "ok", "parsed": {"verdict": v, "reason": "Offline known verdict"}} for v in (1, 0, 1)]
        config = CallConfig()
        result = evaluate({"question": "q", "gold_answer": "gold", "conversation_history": [{"role": "user", "utterance": "prior"}]},
                          [{"text": str(i)} for i in range(3)], caller, config, "test")
        self.assertAlmostEqual(result["score"], (1 + 2 / 3) / 2)
        self.assertEqual(caller.call.call_count, 3)
        for call in caller.call.call_args_list:
            self.assertEqual(call.args[2], config)
            self.assertIn("prior", call.args[1][0]["content"])
            self.assertIn("gold", call.args[1][0]["content"])

    def test_failed_verdict_stops_without_ragas_retry_or_partial_zero(self):
        caller = Mock()
        caller.call.side_effect = [{"status": "ok", "parsed": {"verdict": 1, "reason": "Offline"}}, {"status": "failed"}]
        result = evaluate({"question": "q", "gold_answer": "gold"}, [{"text": str(i)} for i in range(3)], caller, CallConfig(), "test")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["score"])
        self.assertEqual(caller.call.call_count, 2)

    def test_empty_retrieval_and_invalid_verdict(self):
        caller = Mock()
        self.assertEqual(evaluate({"question": "q", "gold_answer": "a"}, [], caller, CallConfig(), "test")["score"], 0)
        caller.call.assert_not_called()
        for value in (True, 2, "1"):
            with self.assertRaises(ValueError):
                validate_verdict({"verdict": value, "reason": "bad"})


class ScorePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "bundle.json"
        self.bundle.write_bytes((RUN / "generation_bundle.json").read_bytes())

    def test_offline_plan_preserves_full_denominator_and_no_calls(self):
        ledger_sha = file_sha256(RUN / "runtime/usage.jsonl")
        with patch("src.evaluation.score_v11.JsonCaller", side_effect=AssertionError("Must not initialize client")):
            plan, bundle, items = prepare(PARENT, RELEASE, self.bundle, ["correctness", "faithfulness"], CallConfig(), 2)
        self.assertEqual(len(bundle["cases"]), 20)
        self.assertEqual(plan["metric_tasks"], 40)
        self.assertEqual(plan["planned_verdict_calls_before_cache"], 40)
        self.assertEqual(file_sha256(RUN / "runtime/usage.jsonl"), ledger_sha)
        baseline = [c for c in bundle["cases"] if c["candidate"] == "rerank_windowed_k5"]
        result = summarize(baseline, items, {}, ["correctness", "faithfulness"])
        self.assertEqual(result["n"], 20)
        self.assertEqual(result["correctness"]["generation_failures"], 0)
        self.assertEqual(result["correctness"]["unknown"], 20)
        self.assertEqual(result["correctness"]["bounds"], [0, 1])

    def test_context_precision_counts_chunks_and_keeps_failed_generations(self):
        plan, bundle, items = prepare(PARENT, RELEASE, self.bundle, ["context_precision"], CallConfig(), 2)
        self.assertEqual(plan["metric_tasks"], 15)  # 15 answer targets, final profile
        self.assertEqual(plan["planned_verdict_calls_before_cache"], 75)  # 15 x 5
        case = copy.deepcopy(next(c for c in bundle["cases"] if items[c["question_id"]]["task_label"] == "answer"))
        case.update(status="failed", response=None)
        result = summarize([case], items, {(case["candidate"], case["question_id"], "context_precision"): {"status": "ok", "score": 1}}, ["context_precision"])
        self.assertEqual(result["context_precision"]["scored_n"], 1)
        self.assertEqual(result["correctness"]["generation_failures"], 1)

    def test_missing_bundle_row_and_bad_input_versions_rejected(self):
        value = json.loads(self.bundle.read_text())
        value["cases"].pop()
        self.bundle.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "omits"):
            load_bundle(PARENT, RELEASE, self.bundle)

    def test_no_claims_and_missing_scores_are_not_perfect(self):
        cases = [{"candidate": "c", "question_id": str(i), "status": "ok"} for i in range(3)]
        items = {str(i): {"task_label": "answer"} for i in range(3)}
        results = {("c", "0", "faithfulness"): {"status": "ok", "score": None},
                   ("c", "1", "faithfulness"): {"status": "ok", "score": 0.5}}
        report = summarize(cases, items, results, ["faithfulness"])
        self.assertEqual(report["faithfulness"]["known_no_fact_n"], 1)
        self.assertEqual(report["faithfulness"]["conservative_bounds_allowing_missing_to_be_fact_bearing"], [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
