"""Agent-selected retrieval for lesson 18."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agentsmith.agent import run_manual_tool_loop, tool_was_used
from agentsmith.models import create_live_model
from agentsmith.rag import make_retrieve_docs_tool


class RagRuleBasedChatModel:
    """Deterministic model double that selects retrieval for policy questions."""

    def __init__(self) -> None:
        self.bound_tool_names: tuple[str, ...] = ()
        self._call_number = 0

    def bind_tools(self, tools):
        self.bound_tool_names = tuple(item.name for item in tools)
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self._call_number += 1
        last = messages[-1]
        if isinstance(last, ToolMessage):
            sources = tuple(dict.fromkeys(re.findall(r"source='([^']+)'", str(last.content))))
            suffix = f"\nSources: {', '.join(sources)}" if sources else ""
            return AIMessage(content=f"Retrieved policy evidence:\n{last.content}{suffix}")

        human = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)), None
        )
        question = str(human.content if human else "").casefold()
        policy_terms = (
            "policy",
            "vacation",
            "holiday",
            "parental",
            "remote",
            "security",
            "device",
            "password",
            "secret",
            "expense",
            "receipt",
            "travel",
            "meal",
            "reimburse",
        )
        if "retrieve_docs" in self.bound_tool_names and any(
            term in question for term in policy_terms
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_docs",
                        "args": {"query": str(human.content)},
                        "id": f"rag-offline-{self._call_number}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Hello! This conversational question needs no policy retrieval.")


def run_agentic_rag(vector_store: Any, question: str, *, live: bool = False):
    retrieve_docs = make_retrieve_docs_tool(vector_store)
    model = create_live_model() if live else RagRuleBasedChatModel()
    messages = run_manual_tool_loop(
        model,
        [HumanMessage(content=question)],
        tools=[retrieve_docs],
    )
    return messages, tool_was_used(messages)
