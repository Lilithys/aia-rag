import json
import tempfile
import unittest
from pathlib import Path

from src.generation.pii import redact_pii, redact_payload
from src.service.runtime import append_event


class PiiTests(unittest.TestCase):
    def test_chinese_adjacent_identifiers_and_format_variants(self):
        cases = ["手机号13800138000", "手机+86 138 0013 8000联系", "身份证11010119900307451X",
                 "身份证11010119900307451x", "电话(518) 474-0774", "SSN为123-45-6789。",
                 "邮箱test.user+qa@example.org请联系", "我的号码是555-123-4567。"]
        for text in cases:
            with self.subTest(text=text):
                redacted = redact_pii(text)
                self.assertNotEqual(text, redacted)
                self.assertNotRegex(redacted, r"[0-9]")
        self.assertEqual(redact_pii(cases[0]), "手机号[PHONE]")
        self.assertEqual(redact_pii(cases[2]), "身份证[ID_NUMBER]")

    def test_ordinary_dates_fees_and_forms_survive(self):
        text = "费用$17.50，60天内提交MV-44；日期2026-09-06，24个月服役，电话说明。"
        self.assertEqual(redact_pii(text), text)
        trace = "a123456789012345678901234567890123456789012345678901234567890123b"
        self.assertEqual(redact_payload({"config_sha256": trace})["config_sha256"], trace)

    def test_nested_values_secrets_and_numeric_metrics(self):
        original = {"Authorization": "Bearer example-secret", "error": {"details": ["手机号13800138000"]},
                    "prompt_tokens": 13800138000, "api-key": "example-key"}
        value = redact_payload(original)
        self.assertEqual(value["Authorization"], "[REDACTED]")
        self.assertEqual(value["api-key"], "[REDACTED]")
        self.assertEqual(value["error"]["details"], ["手机号[PHONE]"])
        self.assertEqual(value["prompt_tokens"], original["prompt_tokens"])
        self.assertEqual(original["Authorization"], "Bearer example-secret")

    def test_logging_redacts_at_persistence_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "requests.jsonl"
            entry = dict(timestamp=0, request_id="req", session_id="session", query_redacted="手机号13800138000",
                retrieval_top1_score=0.9, retrieved_chunk_ids=[], retrieved_doc_ids=[], refused=False,
                refusal_reason=None, answer_redacted="邮箱test@example.org", citations=[], citations_valid=True,
                token_usage={"prompt_tokens": 100, "details": {"api_key": "example-key", "error": "身份证11010119900307451X"}})
            append_event(path, entry)
            saved = json.loads(path.read_text())
            self.assertNotIn("13800138000", path.read_text())
            self.assertNotIn("11010119900307451X", path.read_text())
            self.assertNotIn("example-key", path.read_text())
            self.assertEqual(saved["token_usage"]["prompt_tokens"], 100)
            self.assertIn("[EMAIL]", saved["answer_redacted"])


if __name__ == "__main__":
    unittest.main()
