"""Explicit reference-condition checks, separately versioned from v1.1 scores."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from src.artifacts import file_sha256, fingerprint
from .paid_calls_v1 import input_upper_bound, price
from .rubric_v11 import correctness_messages as previous_messages
from .score_v11 import prepare as previous_prepare

ADDITION = """
Correctness judge revision 1.3 (the dataset and reference are unchanged):
First identify the essential reference facts/conditions that answer the actual question in its history. Explicit acceptance_criteria take precedence. Do not require incidental wording, politeness, or an original dialogue act; supported conditional guidance is valid for a clarification task.
For answer tasks, a related explanation is not sufficient if it omits the requested fact, an essential eligibility condition, a necessary deadline, exception or restriction. Apply the SAME standard regardless of wording or length. Do not assume an omitted condition is implied by a related label. Do not treat instructions for a different branch/action as fulfilling the requested branch. Judge semantic paraphrases fairly; never demand verbatim reference text.
Audit essential conditions as checks. requirement is a short semantic summary of a necessary condition supported by the supplied reference, its source blocks, or explicit acceptance_criteria; it is NOT a verbatim quotation. Do not reject a faithful paraphrase or translation merely for wording differences. response_quote must be an exact response substring for covered/contradicted, and an empty string for missing. Include all essential conditions, not only those covered. A reason describing a requirement as absent must have coverage='missing', never 'covered'. Response quotes are audit anchors, not a lexical matching score. For clarify/refuse, checks can be empty when the behavioral requirement suffices. For answer, at least one check is required. Do not invent a denial: missing information is missing, not contradicted.
behavior_ok assesses task/branch relevance and appropriate clarification/refusal. extra_claims_supported assesses all additional factual claims against retrieved contexts. correct must equal behavior_ok AND extra_claims_supported AND every check having coverage='covered'. An incomplete but grounded answer can have high faithfulness and still be incorrect.
Replace the earlier output schema with ONLY JSON:
{"correct": true|false, "observed_action": "answer|clarify|refuse", "behavior_ok": true|false, "extra_claims_supported": true|false, "checks": [{"requirement": "necessary condition summary from supplied reference/criteria", "coverage": "covered|missing|contradicted", "response_quote": "exact response substring or empty", "reason": "brief semantic explanation"}], "reason": "brief overall explanation"}.
"""


def messages(item, response, contexts):
    result = previous_messages(item, response, contexts)
    result[0]["content"] += ADDITION
    return result


def validate(value, item, response):
    fields = {"correct", "observed_action", "behavior_ok", "extra_claims_supported", "checks", "reason"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Wrong condition-coverage schema")
    if any(type(value[k]) is not bool for k in ("correct", "behavior_ok", "extra_claims_supported")):
        raise ValueError("Expected boolean decisions")
    if value["observed_action"] not in ("answer", "clarify", "refuse") or not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Invalid action/reason")
    checks = value["checks"]
    if not isinstance(checks, list) or (item["task_label"] == "answer" and not checks):
        raise ValueError("Missing essential reference checks")
    seen = set()
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"requirement", "coverage", "response_quote", "reason"}:
            raise ValueError("Invalid reference check")
        quote, answer = check["requirement"], check["response_quote"]
        if not isinstance(quote, str) or not quote.strip() or quote in seen:
            raise ValueError("Empty/duplicate requirement summary")
        seen.add(quote)
        if check["coverage"] not in ("covered", "missing", "contradicted") or not isinstance(answer, str):
            raise ValueError("Invalid coverage")
        if check["coverage"] == "missing":
            if answer:
                raise ValueError("Missing requirement must not invent a response quote")
        elif not answer.strip() or answer not in response["answer"]:
            raise ValueError("Invented response quote")
        if not isinstance(check["reason"], str) or not check["reason"].strip():
            raise ValueError("Missing check reason")
    expected = value["behavior_ok"] and value["extra_claims_supported"] and all(c["coverage"] == "covered" for c in checks)
    if value["correct"] != expected:
        raise ValueError("Correctness conflicts with component checks")


def prepare(parent, release, bundle_path, config, cap):
    plan, bundle, items = previous_prepare(parent, release, bundle_path, ["correctness"], config, cap)
    by_key = {(c["candidate"], c["question_id"]): c for c in bundle["cases"]}
    jobs = []
    for old in plan["tasks"]:
        case = by_key[(old["candidate"], old["question_id"])]
        task = {k: v for k, v in old.items() if k != "task_id"}
        task["messages"] = [messages(items[old["question_id"]], case["response"], case["contexts"])]
        jobs.append({"task_id": fingerprint(task), **task})
    plan["tasks"] = jobs
    binding = plan["binding"]
    binding["runner"] = "score-correctness-1.3"
    binding["code_sha256"]["src/evaluation/correctness_v13.py"] = file_sha256(Path(__file__))
    binding["task_ids"] = [j["task_id"] for j in jobs]
    plan["binding_sha256"] = fingerprint(binding)
    unique = {fingerprint(j["messages"][0]) for j in jobs}
    plan["unique_verdict_inputs"] = len(unique)
    plan["conservative_peak_byte_bound_rmb"] = sum(price(config.model, input_upper_bound(j["messages"][0]), config.max_tokens) * config.max_attempts for j in jobs)
    plan["input_limit_exceeding_calls"] = sum(input_upper_bound(j["messages"][0]) > config.input_token_upper_limit for j in jobs)
    return plan, bundle, items


def score_task(job, case, item, caller, config, role):
    validator = lambda value: validate(value, item, case["response"])
    call = caller.call(role, job["messages"][0], config, validator)
    result = {"status": call["status"], "calls": [call], "score": None}
    if call["status"] == "ok":
        result.update(score=int(call["parsed"]["correct"]), verdict=call["parsed"])
    return result
