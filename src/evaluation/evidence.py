"""Conservative, position-based annotation coverage; no fuzzy correctness match.

Canonical source blocks are aligned to independently parsed/cleaned documents.
Chunks are then aligned to those documents using exact character runs. Missing
source blocks remain missing, including OCR corruption of a single digit.
Annotation text/IDs are never inserted into retrieval or generation inputs.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict


ALIGNMENT_VERSION = "canonical-exact-source-blocks-v3"


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    # The dataset flattens lists inside source blocks while its Markdown
    # renderer adds bullets. Remove textual list markers without erasing
    # numeric minus signs, decimal points or ranges such as June - July.
    text = re.sub(r"(?m)^[ \t]*[-*•]\s+(?=[A-Za-z\u3400-\u9fff])", "", text)
    text = re.sub(r"(?<=[。.;:：])\s*[-*•]\s+(?=[A-Za-z\u3400-\u9fff])", "", text)
    # Remove presentation markers/whitespace only. Digits, words, negation,
    # decimal points, currency signs, percentages and URL text are preserved.
    return re.sub(r"[\s#*`\\\[\]]+", "", text).lower()


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def aligned_blocks(body: str, blocks: list[dict]) -> tuple[str, list[dict], dict]:
    document = canonical(body)
    cursor = 0
    aligned = []
    missing = []
    for block in blocks:
        text = canonical(block["text"])
        if not text:
            missing.append(block["source_block_id"])
            continue
        start = document.find(text, cursor)
        if start < 0:
            start = document.find(text)
        if start < 0:
            missing.append(block["source_block_id"])
            continue
        end = start + len(text)
        aligned.append({"source_block_id": block["source_block_id"], "start": start, "end": end,
                        "block_length": len(text), "location": block.get("location", {})})
        cursor = end
    return document, aligned, {"blocks": len(blocks), "aligned": len(aligned), "missing_ids": missing}


def attach_provenance(chunks, body: str, blocks: list[dict]) -> dict:
    document, aligned, report = aligned_blocks(body, blocks)
    for chunk in chunks:
        text = chunk.text
        if chunk.strategy == "semantic_section" and text.startswith("[") and "\n" in text:
            text = text.split("\n", 1)[1]  # generated breadcrumb has no document-body offsets
        text = canonical(text)
        start = document.find(text) if text else -1
        if start >= 0:
            runs = [(start, start + len(text))]
        else:
            # Structural chunkers may assemble non-adjacent lead-in + section
            # text. Only exact contiguous runs count; unmatched text never does.
            matches = difflib.SequenceMatcher(None, document, text, autojunk=False).get_matching_blocks()
            runs = [(m.a, m.a + m.size) for m in matches if m.size >= 16]
        chunk.source_spans = []
        visible_headings = {canonical(h) for h in chunk.heading_path if canonical(h) in canonical(chunk.text)} if chunk.strategy == "semantic_section" else set()
        for block in aligned:
            intersections = merge_intervals((max(left, block["start"]), min(right, block["end"])) for left, right in runs)
            if document[block["start"]:block["end"]] in visible_headings:
                # A parser-derived heading copied into the actual breadcrumb
                # is visible to encoder/generator even when outside body runs.
                intersections = [[block["start"], block["end"]]]
            for left, right in intersections:
                chunk.source_spans.append({
                    "source_block_id": block["source_block_id"],
                    "start": left - block["start"], "end": right - block["start"],
                    "block_length": block["block_length"], "location": block["location"],
                })
    return report


def answer_targets(item: dict) -> set[tuple[str, str]]:
    return {(e["doc_id"], s["source_block_id"])
            for e in item.get("evidence", []) for s in e.get("spans", [])
            if s.get("source_turn") == "answer" and s.get("source_block_id")}


def covered_targets(chunks, targets) -> set[tuple[str, str]]:
    spans = defaultdict(list)
    lengths = {}
    for chunk in chunks:
        for span in chunk.source_spans:
            key = (chunk.doc_id, span["source_block_id"])
            if key not in targets:
                continue
            if not 0 <= span["start"] < span["end"] <= span["block_length"]:
                raise ValueError("invalid source interval")
            if key in lengths and lengths[key] != span["block_length"]:
                raise ValueError("inconsistent source block length")
            lengths[key] = span["block_length"]
            spans[key].append((span["start"], span["end"]))
    return {key for key, intervals in spans.items()
            if sum(end - start for start, end in merge_intervals(intervals)) == lengths[key]}


def score_window(chunks, targets) -> dict:
    if not targets:
        raise ValueError("source retrieval target lacks answer-side annotations")
    covered = covered_targets(chunks, targets)
    return {
        "any_answer_evidence_hit": int(bool(covered)),
        "annotated_answer_block_recall": len(covered) / len(targets),
        "all_annotated_answer_blocks_hit": int(covered == targets),
        "annotated_evidence_precision": sum(bool(covered_targets([c], targets)) for c in chunks) / len(chunks) if chunks else 0.0,
        "context_tokens": sum(c.token_count for c in chunks),
        "chunks_returned": len(chunks),
    }


def retrieval_query(item: dict) -> str:
    turns = [f"{t['role']}: {t['utterance']}" for t in item.get("conversation_history", [])]
    return "\n".join(turns + [f"user: {item['question']}"])
