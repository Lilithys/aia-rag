"""RAGAS 0.3.1 Context Precision through the audited JSON transport.

Use the installed metric and its unchanged prompt/aggregation. Each context
requires one verdict. RAGAS cannot add unaccounted retries or repair calls.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from langchain_core.outputs import Generation, LLMResult
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics._context_precision import ContextPrecisionPrompt, LLMContextPrecisionWithReference, QAC

from .evidence import retrieval_query


def check_version():
    if importlib.metadata.version("ragas") != "0.3.1":
        raise ValueError("This adapter requires the pinned RAGAS 0.3.1")


def messages_for(item, contexts):
    check_version()
    prompt = ContextPrecisionPrompt()
    return [[{"role": "user", "content": prompt.to_string(QAC(question=retrieval_query(item),
                answer=item["gold_answer"], context=context["text"]))}] for context in contexts]


def validate_verdict(value):
    if not isinstance(value, dict) or set(value) != {"reason", "verdict"}:
        raise ValueError("Invalid Context Precision verdict schema")
    if type(value["verdict"]) is not int or value["verdict"] not in (0, 1):
        raise ValueError("Context Precision verdict must be integer 0 or 1")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Context Precision requires an explanation")


class MetricCallFailed(RuntimeError):
    pass


class BudgetedRagasLLM(BaseRagasLLM):
    def __init__(self, caller, config, role, expected_messages):
        super().__init__()
        self.caller, self.config, self.role = caller, config, role
        self.expected_messages, self.calls = expected_messages, []

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        messages = [{"role": "user", "content": prompt.to_string()}]
        index = len(self.calls)
        if n != 1 or stop or index >= len(self.expected_messages) or messages != self.expected_messages[index]:
            raise ValueError("Unplanned RAGAS call or repair attempt")
        # Fixed CallConfig controls temperature; RAGAS defaults cannot override it.
        result = self.caller.call(self.role, messages, self.config, validate_verdict)
        self.calls.append(result)
        if result["status"] != "ok":
            raise MetricCallFailed(result["status"])
        validate_verdict(result["parsed"])
        return LLMResult(generations=[[Generation(text=json.dumps(result["parsed"]))]])

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return self.generate_text(prompt, n, temperature, stop, callbacks)

    async def generate(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        # Override BaseRagasLLM.generate: its implicit async retry wrapper is
        # intentionally bypassed. Only JsonCaller's bounded attempts apply.
        return await self.agenerate_text(prompt, n, temperature, stop, callbacks)


def evaluate(item, contexts, caller, config, role):
    check_version()
    if not contexts:
        return {"status": "ok", "score": 0.0, "calls": [], "empty_retrieval": True}
    llm = BudgetedRagasLLM(caller, config, role, messages_for(item, contexts))
    metric = LLMContextPrecisionWithReference(llm=llm)
    sample = SingleTurnSample(user_input=retrieval_query(item), reference=item["gold_answer"],
                              retrieved_contexts=[c["text"] for c in contexts])
    try:
        # The public wrapper also creates a telemetry event (and accesses a
        # user-ID file even with tracking disabled). Invoke its unchanged
        # metric implementation directly; input columns and callbacks here
        # are explicit, and no external telemetry state is needed.
        score = asyncio.run(metric._single_turn_ascore(sample, callbacks=[]))
        return {"status": "ok", "score": float(score), "calls": llm.calls, "empty_retrieval": False}
    except MetricCallFailed as exc:
        return {"status": str(exc), "score": None, "calls": llm.calls}
