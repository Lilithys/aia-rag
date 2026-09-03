"""Step 2a: clean & prepare the step-1 Markdown output for chunking.

Reads data/processed/markdown/*.md (frontmatter + body), cleans the body,
and writes data/interim/markdown_clean/*.md with the same frontmatter.

Usage:
    uv run python -m src.chunking.prepare
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

from .clean import clean_body

_FRONTMATTER_RE = re.compile(r"\A(---\n.*?\n---\n)(.*)", re.DOTALL)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="data/processed/markdown")
    parser.add_argument("--out-dir", default="data/interim/markdown_clean")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.md")))

    total_dropped = 0
    total_escapes = 0
    per_doc = []
    for path in files:
        text = open(path, encoding="utf-8").read()
        m = _FRONTMATTER_RE.match(text)
        if not m:
            print(f"WARNING: no frontmatter in {path}, skipping", file=sys.stderr)
            continue
        frontmatter, body = m.group(1), m.group(2)
        cleaned, stats = clean_body(body)
        out_path = os.path.join(args.out_dir, os.path.basename(path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.write("\n")
            f.write(cleaned)
        total_dropped += stats["boilerplate_sections_dropped"]
        total_escapes += stats["escape_sequences_fixed"]
        per_doc.append({"file": os.path.basename(path), **stats})

    print(f"Cleaned {len(files)} documents -> {args.out_dir}", file=sys.stderr)
    print(f"  boilerplate sections dropped: {total_dropped}", file=sys.stderr)
    print(f"  escape sequences fixed: {total_escapes}", file=sys.stderr)

    report_path = os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "cleaning_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"total_boilerplate_dropped": total_dropped, "total_escapes_fixed": total_escapes, "documents": per_doc},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Report written to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
