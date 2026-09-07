import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.service.http_api import SessionRegistry, handler_for
from src.service.runtime import RagRuntime, ServiceConfig
from http.server import ThreadingHTTPServer


class FakeRetriever:
    def retrieve(self, _item, _candidate):
        return [], {"top1_dense_score": 0.0}


class FakeCaller:
    def call(self, *_args):
        raise AssertionError("low confidence must refuse before generation")


class HttpApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        runtime = RagRuntime(FakeRetriever(), FakeCaller(), ServiceConfig(), Path(self.temp.name) / "requests.jsonl")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(SessionRegistry(runtime)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as exc:
            response = exc
        return response.status, json.loads(response.read())

    def test_health_and_refusal_contract(self):
        status, health = self.request("/health")
        self.assertEqual((status, health["status"]), (200, "ok"))
        status, result = self.request("/ask", {"question": "今天的天气？"})
        self.assertEqual((status, result["status"], result["action"]), (200, "ok", "refuse"))
        self.assertEqual(result["citations"], [])
        self.assertTrue(result["request_id"])

    def test_session_reset_rotates_identifier(self):
        _, first = self.request("/ask", {"question": "Weather?"})
        status, reset = self.request("/sessions/reset", {"session_id": first["session_id"]})
        self.assertEqual(status, 200)
        self.assertNotEqual(first["session_id"], reset["session_id"])
        status, error = self.request("/ask", {"question": "again", "session_id": first["session_id"]})
        self.assertEqual((status, error["error"]), (404, "unknown_session"))

    def test_bad_json_and_unknown_fields_rejected(self):
        status, error = self.request("/ask", {"question": "x", "extra": True})
        self.assertEqual((status, error["error"]), (400, "unsupported_field"))


if __name__ == "__main__":
    unittest.main()
