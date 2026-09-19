"""Small transparent evaluation runner used by lesson 10."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agentsmith.agent import build_agent_graph, invoke_agent, tool_was_used
from agentsmith.models import select_model


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    answer_correct: bool
    tool_use_correct: bool
    answer: str

    @property
    def passed(self) -> bool:
        return self.answer_correct and self.tool_use_correct


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_cases(cases: list[dict[str, Any]], *, live: bool = False) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for case in cases:
        graph = build_agent_graph(select_model(live=live))
        state = invoke_agent(graph, [HumanMessage(content=case["question"])])
        answer = str(state["messages"][-1].content)
        used_tool = tool_was_used(state["messages"])
        results.append(
            EvaluationResult(
                case_id=case["id"],
                answer_correct=case["expected_contains"].casefold() in answer.casefold(),
                tool_use_correct=used_tool is case["expected_tool_use"],
                answer=answer,
            )
        )
    return results
