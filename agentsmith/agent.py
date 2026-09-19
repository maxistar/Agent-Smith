"""Manual and LangGraph implementations of the same tool-calling loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

from agentsmith.tools import search_docs

DEFAULT_TOOLS: tuple[BaseTool, ...] = (search_docs,)


class AgentLoopLimitError(RuntimeError):
    """Raised when a model keeps requesting tools past a safe limit."""


def tool_registry(tools: Iterable[BaseTool] = DEFAULT_TOOLS) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}


def bind_tools(model: Any, tools: Sequence[BaseTool] = DEFAULT_TOOLS):
    """Bind schemas to a live or deterministic model."""
    return model.bind_tools(list(tools))


def execute_tool_calls(
    message: AIMessage,
    tools: Mapping[str, BaseTool] | None = None,
) -> list[ToolMessage]:
    """Execute requested tools and convert all failures into inspectable messages."""
    registry = dict(tools or tool_registry())
    results: list[ToolMessage] = []

    for call in message.tool_calls:
        name = call.get("name", "")
        call_id = call.get("id") or "missing-tool-call-id"
        arguments = call.get("args", {})
        selected = registry.get(name)
        if selected is None:
            results.append(
                ToolMessage(
                    content=f"ERROR: Unknown tool {name!r}.",
                    tool_call_id=call_id,
                    name=name or None,
                    status="error",
                )
            )
            continue

        try:
            output = selected.invoke(arguments)
        except Exception as exc:  # educational boundary: make tool failures observable
            results.append(
                ToolMessage(
                    content=f"ERROR: {type(exc).__name__}: {exc}",
                    tool_call_id=call_id,
                    name=name,
                    status="error",
                )
            )
        else:
            results.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=call_id,
                    name=name,
                    status="success",
                )
            )
    return results


def run_manual_tool_loop(
    model: Any,
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[BaseTool] = DEFAULT_TOOLS,
    max_iterations: int = 4,
) -> list[BaseMessage]:
    """Run the tool loop explicitly so every application-side step is visible."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    bound_model = bind_tools(model, tools)
    history = list(messages)
    registry = tool_registry(tools)

    for _ in range(max_iterations):
        response = bound_model.invoke(history)
        if not isinstance(response, AIMessage):
            raise TypeError("The chat model must return an AIMessage.")
        history.append(response)
        if not response.tool_calls:
            return history
        history.extend(execute_tool_calls(response, registry))

    raise AgentLoopLimitError(
        f"Agent stopped after {max_iterations} model iterations with tool calls still pending."
    )


def build_agent_graph(
    model: Any,
    *,
    tools: Sequence[BaseTool] = DEFAULT_TOOLS,
    checkpointer: Any | None = None,
):
    """Compile the low-level graph equivalent of ``run_manual_tool_loop``."""
    bound_model = bind_tools(model, tools)
    registry = tool_registry(tools)

    def call_model(state: MessagesState) -> dict[str, list[AIMessage]]:
        response = bound_model.invoke(state["messages"])
        if not isinstance(response, AIMessage):
            raise TypeError("The chat model must return an AIMessage.")
        return {"messages": [response]}

    def call_tools(state: MessagesState) -> dict[str, list[ToolMessage]]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            raise TypeError("The tools node expects the latest message to be an AIMessage.")
        return {"messages": execute_tool_calls(last, registry)}

    def route_after_model(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_model, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def invoke_agent(
    graph: Any,
    messages: Sequence[BaseMessage],
    *,
    thread_id: str | None = None,
    recursion_limit: int = 12,
) -> MessagesState:
    """Invoke a graph with a learner-readable recursion-limit error."""
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    try:
        return graph.invoke({"messages": list(messages)}, config=config)
    except GraphRecursionError as exc:
        raise AgentLoopLimitError(
            f"Agent exceeded the graph recursion limit ({recursion_limit})."
        ) from exc


def format_trajectory(messages: Sequence[BaseMessage]) -> str:
    """Render message types and tool calls without hiding intermediate state."""
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        label = type(message).__name__
        if isinstance(message, AIMessage) and message.tool_calls:
            detail = f"tool_calls={message.tool_calls}"
        else:
            detail = str(message.content)
        lines.append(f"{index}. {label}: {detail}")
    return "\n".join(lines)


def tool_was_used(messages: Sequence[BaseMessage]) -> bool:
    return any(isinstance(message, ToolMessage) for message in messages)
