"""Preprocessing entrypoint: converts every document in the manifest to
Markdown, preserving heading structure, and writes one .md file per
document plus a JSON run report.

Usage:
    uv run python -m src.preprocess.run
    uv run python -m src.preprocess.run --doc-root doc --out-dir data/processed/markdown
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from . import convert_docx, convert_markdown, convert_pdf_scanned, convert_pdf_text, convert_txt
from .common import make_frontmatter, normalize_blank_lines, word_count
from .manifest import DocEntry, load_manifest

CONVERTERS = {
    "markdown": ("native", convert_markdown.convert),
    "txt": ("txt-heuristic", convert_txt.convert),
    "docx": ("docx-styles", convert_docx.convert),
    "pdf_text": ("pdf-text-extraction", convert_pdf_text.convert),
    "pdf_scanned": ("ocr-tesseract", convert_pdf_scanned.convert),
}


def process_one(doc: DocEntry, doc_root: str, out_dir: str) -> dict:
    src_path = os.path.join(doc_root, doc.file)
    conversion_method, converter = CONVERTERS[doc.format]

    t0 = time.time()
    if doc.format == "pdf_scanned":
        body = converter(src_path, language=doc.language)
    else:
        body = converter(src_path)
    body = normalize_blank_lines(body)
    elapsed = time.time() - t0

    out_name = os.path.splitext(os.path.basename(doc.file))[0] + ".md"
    out_path = os.path.join(out_dir, out_name)

    meta = {
        "doc_id": doc.doc_id,
        "domain": doc.domain,
        "title": doc.title,
        "language": doc.language,
        "format": doc.format,
        "source_file": doc.file,
        "conversion_method": conversion_method,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(make_frontmatter(meta))
        f.write(body)

    return {
        "doc_id": doc.doc_id,
        "format": doc.format,
        "source_file": doc.file,
        "output_file": os.path.relpath(out_path, out_dir),
        "conversion_method": conversion_method,
        "word_count": word_count(body),
        "heading_count": sum(1 for line in body.splitlines() if line.startswith("#")),
        "seconds": round(elapsed, 3),
        "status": "ok",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-root", default="doc", help="root of the source dataset (default: doc)")
    parser.add_argument(
        "--out-dir",
        default="data/processed/markdown",
        help="output directory for converted Markdown (default: data/processed/markdown)",
    )
    parser.add_argument("--formats", nargs="*", default=None, help="only process these source formats")
    parser.add_argument("--doc-ids", nargs="*", default=None, help="only process these doc_ids")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    docs = load_manifest(args.doc_root)
    if args.formats:
        docs = [d for d in docs if d.format in args.formats]
    if args.doc_ids:
        wanted = set(args.doc_ids)
        docs = [d for d in docs if d.doc_id in wanted]

    report = []
    errors = []
    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc.format:12s} {doc.file}", file=sys.stderr)
        try:
            report.append(process_one(doc, args.doc_root, args.out_dir))
        except Exception as e:  # noqa: BLE001 - we want to keep going and report failures
            traceback.print_exc()
            errors.append(
                {
                    "doc_id": doc.doc_id,
                    "format": doc.format,
                    "source_file": doc.file,
                    "error": f"{type(e).__name__}: {e}",
                    "status": "failed",
                }
            )

    summary = {
        "total_documents": len(docs),
        "succeeded": len(report),
        "failed": len(errors),
        "by_format": {},
        "documents": report + errors,
    }
    for fmt in CONVERTERS:
        fmt_entries = [r for r in report if r["format"] == fmt]
        summary["by_format"][fmt] = {
            "count": len(fmt_entries),
            "total_words": sum(r["word_count"] for r in fmt_entries),
        }

    report_path = os.path.join(os.path.dirname(args.out_dir.rstrip("/")) or ".", "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nConverted {summary['succeeded']}/{summary['total_documents']} documents.", file=sys.stderr)
    if errors:
        print(f"{len(errors)} FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e['doc_id']} ({e['source_file']}): {e['error']}", file=sys.stderr)
    print(f"Report written to {report_path}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
