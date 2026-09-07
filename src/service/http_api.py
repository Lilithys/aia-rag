"""Small JSON HTTP API over the same budgeted runtime used by the CLI demo."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from dotenv import load_dotenv

from src.service.final_runtime import make_runtime
from src.service.runtime import ServiceConfig, Session
from src.paths import FINAL_INDEX


MAX_REQUEST_BYTES = 64 * 1024


class SessionRegistry:
    def __init__(self, runtime):
        self.runtime = runtime
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def ask(self, question, session_id=None):
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError("session_id_must_be_nonempty_string")
        with self._lock:
            if session_id is None:
                session = Session()
                self._sessions[session.session_id] = session
            else:
                session = self._sessions.get(session_id)
                if session is None:
                    raise KeyError("unknown_session")
            result = session.ask(self.runtime, question)
            return session.session_id, result

    def reset(self, session_id):
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise KeyError("unknown_session")
            session.reset()
            self._sessions[session.session_id] = session
            return session.session_id


def response_payload(session_id, result):
    response = result.get("response") or {}
    return {
        "session_id": session_id,
        "request_id": result["request_id"],
        "status": result["status"],
        "answer": response.get("answer"),
        "action": response.get("action"),
        "citations": result.get("citations", []),
        "error_type": result.get("error_type"),
        "latency_ms": result.get("latency_ms", {}),
    }


def handler_for(registry: SessionRegistry):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AiaRag/1.0"

        def log_message(self, _format, *_args):
            return

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid_content_length") from exc
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid_request_size")
            try:
                value = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid_json") from exc
            if not isinstance(value, dict):
                raise ValueError("json_object_required")
            return value

        def do_GET(self):
            if self.path != "/health":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self.send_json(HTTPStatus.OK, {
                "status": "ok",
                "run_id": registry.runtime.run_id,
                "config_sha256": registry.runtime.config_sha256,
            })

        def do_POST(self):
            try:
                body = self.read_json()
                if self.path == "/ask":
                    if set(body) - {"question", "session_id"}:
                        raise ValueError("unsupported_field")
                    session_id, result = registry.ask(body.get("question"), body.get("session_id"))
                    status = HTTPStatus.OK if result["status"] == "ok" else HTTPStatus.SERVICE_UNAVAILABLE
                    self.send_json(status, response_payload(session_id, result))
                elif self.path == "/sessions/reset":
                    if set(body) != {"session_id"} or not isinstance(body["session_id"], str):
                        raise ValueError("session_id_required")
                    self.send_json(HTTPStatus.OK, {"session_id": registry.reset(body["session_id"]), "status": "reset"})
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except KeyError as exc:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": exc.args[0]})
            except ValueError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--artifact-folder", type=Path, default=FINAL_INDEX)
    parser.add_argument("--state-dir", type=Path, default=Path("data/logs/http_service"))
    parser.add_argument("--candidate", choices=("baseline_dense_k10", "rerank_truncated_k10", "rerank_windowed_k5"), default="rerank_windowed_k5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--budget-rmb", type=float, default=2.0)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    load_dotenv(args.env_file, override=False)
    config = ServiceConfig(candidate=args.candidate, temperature=args.temperature)
    runtime, _ledger = make_runtime(args.artifact_folder, args.state_dir, config, args.budget_rmb)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(SessionRegistry(runtime)))
    print(json.dumps({"status": "ready", "url": f"http://{args.host}:{server.server_port}",
                      "run_id": runtime.run_id, "config_sha256": runtime.config_sha256}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
