"""Live and deterministic chat-model configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5-mini"


class ModelConfigurationError(RuntimeError):
    """Raised when a live lesson is missing safe model configuration."""


def create_live_model():
    """Create the documented live model without exposing credential values."""
    provider = os.getenv("AGENTSMITH_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider != "openai":
        raise ModelConfigurationError(
            f"Unsupported AGENTSMITH_PROVIDER={provider!r}. This learning path currently "
            "documents the 'openai' provider."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise ModelConfigurationError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env, load it in your shell, "
            "and set a real key; never commit the key."
        )

    from langchain_openai import ChatOpenAI

    model_name = os.getenv("AGENTSMITH_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return ChatOpenAI(model=model_name, temperature=0)


def require_langsmith_configuration() -> None:
    """Validate tracing variables without printing their values."""
    enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        raise ModelConfigurationError(
            "LANGSMITH_TRACING is not enabled. Set it to 'true' only for the tracing lesson."
        )
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ModelConfigurationError(
            "LANGSMITH_API_KEY is missing. Add it to your local environment; never commit it."
        )


ResponseFactory = Callable[[Sequence[BaseMessage]], AIMessage]


class ScriptedChatModel:
    """A tiny deterministic model double used by examples and offline tests."""

    def __init__(self, responses: Sequence[AIMessage | ResponseFactory]) -> None:
        if not responses:
            raise ValueError("At least one scripted response is required.")
        self._responses = list(responses)
        self._index = 0
        self.bound_tool_names: tuple[str, ...] = ()

    def bind_tools(self, tools):
        self.bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        if self._index >= len(self._responses):
            raise RuntimeError("The scripted model has no response left for this invocation.")
        response = self._responses[self._index]
        self._index += 1
        return response(messages) if callable(response) else response


class RuleBasedChatModel:
    """A deterministic tool-calling model for safe, network-free lesson runs."""

    def __init__(self) -> None:
        self._call_number = 0
        self.bound_tool_names: tuple[str, ...] = ()

    def bind_tools(self, tools):
        self.bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self._call_number += 1
        last = messages[-1]

        if isinstance(last, ToolMessage):
            return AIMessage(content=f"According to the documentation: {last.content}")

        human = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )
        question = str(human.content if human else "").lower()
        topic = next(
            (
                candidate
                for candidate in ("vacation", "remote", "parental", "dress code")
                if candidate in question
            ),
            None,
        )
        if topic and "search_docs" in self.bound_tool_names:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_docs",
                        "args": {"query": topic},
                        "id": f"offline-call-{self._call_number}",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content="Hello! I can answer this without using a tool.")


def select_model(*, live: bool):
    """Return a paid/networked model only when a learner explicitly opts in."""
    return create_live_model() if live else RuleBasedChatModel()
