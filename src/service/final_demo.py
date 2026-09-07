"""Interactive delivery CLI using a pinned artifact folder and budget."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from dotenv import load_dotenv

from src.service.final_runtime import make_runtime, preflight
from src.service.runtime import ServiceConfig, Session
from src.paths import FINAL_INDEX


DEFAULT_FOLDER = FINAL_INDEX


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--artifact-folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--state-dir", type=Path, default=Path("data/logs/final_cli"))
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--candidate", choices=("baseline_dense_k10", "rerank_truncated_k10", "rerank_windowed_k5"), default="rerank_windowed_k5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--budget-rmb", type=float, default=2.0)
    args = parser.parse_args()
    config = ServiceConfig(candidate=args.candidate, temperature=args.temperature)
    print(json.dumps(preflight(args.artifact_folder, config), ensure_ascii=False, indent=2))
    if not args.run:
        return 0
    load_dotenv(args.env_file, override=False)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime, ledger = make_runtime(args.artifact_folder, args.state_dir, config, args.budget_rmb)
        session = Session()
        print("就绪。输入中英文问题；/reset 清空会话，/exit 退出。")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question in {"/exit", "exit", "quit"}:
                break
            if question == "/reset":
                session.reset()
                print("会话已重置。")
                continue
            if not question:
                continue
            result = session.ask(runtime, question)
            if result["status"] != "ok":
                print(f"请求未完成：{result['error_type'] or result['status']}；request_id={result['request_id']}")
                continue
            print(result["response"]["answer"])
            for citation in result["citations"]:
                print(f"  来源：{citation['document']} / {' > '.join(citation['heading'])} [{citation['chunk_id']}]")
            print(f"耗时 {result['latency_ms']['total'] / 1000:.2f}s；累计计费/保留估计 ¥{ledger.spent_upper():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
