import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agentsmith.agent import AgentLoopLimitError, build_agent_graph, invoke_agent
from agentsmith.models import ModelConfigurationError, ScriptedChatModel, create_live_model


def repeated_request(index: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_docs",
                "args": {"query": "vacation"},
                "id": f"repeat-{index}",
                "type": "tool_call",
            }
        ],
    )


def test_graph_repeated_tool_requests_are_bounded() -> None:
    model = ScriptedChatModel([repeated_request(index) for index in range(10)])
    graph = build_agent_graph(model)
    with pytest.raises(AgentLoopLimitError, match="recursion limit"):
        invoke_agent(graph, [HumanMessage(content="keep going")], recursion_limit=4)


def test_missing_credentials_error_does_not_expose_secret(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelConfigurationError) as captured:
        create_live_model()
    message = str(captured.value)
    assert "OPENAI_API_KEY" in message
    assert "replace-me" not in message
