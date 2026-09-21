from langchain_core.messages import AIMessage, ToolMessage

from agentsmith.rag import ABSTENTION, answer_with_rag, build_in_memory_store
from agentsmith.rag_agent import run_agentic_rag
from agentsmith.rag_graph import build_rag_graph


def test_two_step_rag_cites_retrieved_source_and_abstains_without_evidence() -> None:
    store = build_in_memory_store()
    supported = answer_with_rag("How much annual holiday do employees get?", store)
    unsupported = answer_with_rag("What is the policy on musical instruments?", store)

    assert "30 paid vacation days" in supported.answer
    assert supported.sources == ("hr/leave-policy.md",)
    assert not supported.abstained
    assert unsupported.answer == ABSTENTION
    assert unsupported.sources == ()
    assert unsupported.abstained


def test_agentic_rag_uses_bounded_retrieval_only_when_needed() -> None:
    store = build_in_memory_store()
    policy_messages, policy_used = run_agentic_rag(store, "What is the lost-device policy?")
    chat_messages, chat_used = run_agentic_rag(store, "Hello, how are you?")

    assert policy_used
    assert any(isinstance(item, ToolMessage) for item in policy_messages)
    request = next(
        item for item in policy_messages if isinstance(item, AIMessage) and item.tool_calls
    )
    result = next(item for item in policy_messages if isinstance(item, ToolMessage))
    assert result.tool_call_id == request.tool_calls[0]["id"]
    assert "security/access-policy.md" in str(result.content)
    assert len(str(result.content)) < 2000
    assert not chat_used
    assert not any(isinstance(item, ToolMessage) for item in chat_messages)


def test_langgraph_rag_exposes_generate_and_abstain_routes() -> None:
    store = build_in_memory_store()
    graph = build_rag_graph(store)
    supported = graph.invoke({"question": "How many vacation days are provided?", "route": []})
    unsupported = graph.invoke(
        {"question": "What is the policy on musical instruments?", "route": []}
    )

    assert supported["route"] == ["retrieve", "grade", "generate"]
    assert supported["sources"] == ["hr/leave-policy.md"]
    assert "30 paid vacation days" in supported["answer"]
    assert unsupported["route"] == ["retrieve", "grade", "abstain"]
    assert unsupported["answer"] == ABSTENTION
    assert unsupported["sources"] == []
