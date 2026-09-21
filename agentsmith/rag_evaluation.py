"""Retrieval-aware evaluation kept separate from generation quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentsmith.rag import ABSTENTION, answer_from_results, retrieve
from agentsmith.rag_agent import run_agentic_rag


@dataclass(frozen=True)
class RagEvaluationResult:
    case_id: str
    retrieval_hit: bool
    grounded: bool
    citations_valid: bool
    abstention_correct: bool
    tool_use_correct: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.retrieval_hit,
                self.grounded,
                self.citations_valid,
                self.abstention_correct,
                self.tool_use_correct,
            )
        )


def load_rag_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_rag_case(
    case: dict[str, Any],
    *,
    answer: str,
    cited_sources: list[str],
    retrieved_sources: list[str],
    tool_used: bool,
) -> RagEvaluationResult:
    expected_sources = set(case["expected_sources"])
    unsupported = case["unsupported"]
    retrieval_hit = not expected_sources or bool(expected_sources & set(retrieved_sources))
    grounded = (
        ABSTENTION.casefold() in answer.casefold()
        if unsupported
        else case["expected_contains"].casefold() in answer.casefold()
    )
    citations_valid = set(cited_sources).issubset(retrieved_sources) and (
        not expected_sources or bool(cited_sources)
    )
    abstention_correct = (ABSTENTION.casefold() in answer.casefold()) is unsupported
    return RagEvaluationResult(
        case_id=case["id"],
        retrieval_hit=retrieval_hit,
        grounded=grounded,
        citations_valid=citations_valid,
        abstention_correct=abstention_correct,
        tool_use_correct=tool_used is case["expected_tool_use"],
    )


def evaluate_rag_cases(vector_store: Any, cases: list[dict[str, Any]]) -> list[RagEvaluationResult]:
    results: list[RagEvaluationResult] = []
    for case in cases:
        question = case["question"]
        retrieved = (
            retrieve(
                vector_store,
                question,
                k=3,
                metadata_filter=case.get("metadata_filter"),
            )
            if case["expected_tool_use"]
            else []
        )
        if case["expected_tool_use"]:
            rag_answer = answer_from_results(question, retrieved)
            answer = rag_answer.answer
            cited_sources = list(rag_answer.sources)
            tool_used = True
        else:
            messages, tool_used = run_agentic_rag(vector_store, question)
            answer = str(messages[-1].content)
            cited_sources = []
        results.append(
            score_rag_case(
                case,
                answer=answer,
                cited_sources=cited_sources,
                retrieved_sources=[item.source for item in retrieved],
                tool_used=tool_used,
            )
        )
    return results
