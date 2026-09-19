from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentsmith.agent import execute_tool_calls, run_manual_tool_loop, tool_registry
from agentsmith.models import ScriptedChatModel
from agentsmith.tools import search_docs


def tool_request(
    name: str = "search_docs",
    args: dict | None = None,
    call_id: str = "call-1",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args if args is not None else {"query": "vacation"},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_manual_loop_associates_tool_result_with_request() -> None:
    model = ScriptedChatModel(
        [tool_request(call_id="matching-id"), AIMessage(content="Final answer")]
    )
    history = run_manual_tool_loop(model, [HumanMessage(content="vacation?")])

    assert isinstance(history[1], AIMessage)
    assert isinstance(history[2], ToolMessage)
    assert history[2].tool_call_id == "matching-id"
    assert history[2].status == "success"
    assert history[-1].content == "Final answer"


def test_unknown_tool_becomes_error_message() -> None:
    result = execute_tool_calls(tool_request(name="missing"))[0]
    assert result.status == "error"
    assert "Unknown tool" in str(result.content)


def test_invalid_arguments_become_error_message() -> None:
    result = execute_tool_calls(tool_request(args={"query": ""}))[0]
    assert result.status == "error"
    assert "non-empty" in str(result.content)


def test_tool_exception_becomes_error_message() -> None:
    @tool
    def explode(reason: str) -> str:
        """Always fail for a deterministic test."""
        raise RuntimeError(reason)

    result = execute_tool_calls(
        tool_request(name="explode", args={"reason": "boom"}),
        tool_registry([search_docs, explode]),
    )[0]
    assert result.status == "error"
    assert "boom" in str(result.content)
