"""Lesson 09: make common tool and loop failures bounded and visible."""

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from agentsmith.agent import (
    AgentLoopLimitError,
    build_agent_graph,
    execute_tool_calls,
    invoke_agent,
    tool_registry,
)
from agentsmith.models import ScriptedChatModel
from agentsmith.tools import search_docs


@tool
def failing_tool(reason: str) -> str:
    """Demonstrate how application-side tool failures become ToolMessages."""
    raise RuntimeError(reason)


def requested(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def main() -> None:
    registry = tool_registry([search_docs, failing_tool])
    cases = [
        requested("missing_tool", {}, "unknown-1"),
        requested("search_docs", {"query": ""}, "invalid-1"),
        requested("failing_tool", {"reason": "planned failure"}, "failure-1"),
    ]
    for case in cases:
        result = execute_tool_calls(case, registry)[0]
        print(f"{result.tool_call_id}: status={result.status} content={result.content}")

    repeated = [requested("search_docs", {"query": "vacation"}, f"loop-{i}") for i in range(8)]
    graph = build_agent_graph(ScriptedChatModel(repeated))
    try:
        invoke_agent(graph, [HumanMessage(content="loop forever")], recursion_limit=4)
    except AgentLoopLimitError as exc:
        print(f"loop-limit: {exc}")


if __name__ == "__main__":
    main()
