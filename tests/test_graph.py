from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentsmith.agent import build_agent_graph, invoke_agent
from agentsmith.models import ScriptedChatModel


def requested_tool() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_docs",
                "args": {"query": "remote"},
                "id": "graph-call",
                "type": "tool_call",
            }
        ],
    )


def test_graph_routes_tool_result_back_to_model() -> None:
    graph = build_agent_graph(
        ScriptedChatModel([requested_tool(), AIMessage(content="Remote answer")])
    )
    state = invoke_agent(graph, [HumanMessage(content="remote?")])

    assert [type(message) for message in state["messages"]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert state["messages"][2].tool_call_id == "graph-call"
    assert state["messages"][-1].content == "Remote answer"


def test_graph_terminates_without_tool_call() -> None:
    graph = build_agent_graph(ScriptedChatModel([AIMessage(content="Direct answer")]))
    state = invoke_agent(graph, [HumanMessage(content="hello")])
    assert len(state["messages"]) == 2
    assert state["messages"][-1].content == "Direct answer"
