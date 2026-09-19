from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentsmith.agent import build_agent_graph, invoke_agent
from agentsmith.models import RuleBasedChatModel


def test_explicit_conversation_state_accumulates_messages() -> None:
    graph = build_agent_graph(RuleBasedChatModel())
    first = invoke_agent(graph, [HumanMessage(content="vacation policy")])
    second = invoke_agent(
        graph,
        [*first["messages"], HumanMessage(content="remote policy")],
    )
    assert len(second["messages"]) > len(first["messages"])
    assert any("remote" in str(message.content).lower() for message in second["messages"])


def test_checkpointer_continues_same_thread_and_isolates_another() -> None:
    graph = build_agent_graph(RuleBasedChatModel(), checkpointer=InMemorySaver())
    first = invoke_agent(
        graph,
        [HumanMessage(content="vacation policy")],
        thread_id="thread-a",
    )
    continued = invoke_agent(
        graph,
        [HumanMessage(content="remote policy")],
        thread_id="thread-a",
    )
    isolated = invoke_agent(
        graph,
        [HumanMessage(content="hello")],
        thread_id="thread-b",
    )

    assert len(continued["messages"]) > len(first["messages"])
    assert len(isolated["messages"]) == 2


def test_new_in_memory_checkpointer_loses_previous_process_state() -> None:
    first_graph = build_agent_graph(RuleBasedChatModel(), checkpointer=InMemorySaver())
    invoke_agent(
        first_graph,
        [HumanMessage(content="vacation policy")],
        thread_id="same-name",
    )

    reconstructed = build_agent_graph(RuleBasedChatModel(), checkpointer=InMemorySaver())
    state = invoke_agent(
        reconstructed,
        [HumanMessage(content="hello")],
        thread_id="same-name",
    )
    assert len(state["messages"]) == 2
