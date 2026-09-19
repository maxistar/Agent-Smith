from langchain_core.messages import HumanMessage

from agentsmith.agent import build_agent_graph, invoke_agent
from agentsmith.models import RuleBasedChatModel


def test_offline_graph_does_not_need_provider_or_tracing_credentials(monkeypatch) -> None:
    for name in ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING"):
        monkeypatch.delenv(name, raising=False)

    state = invoke_agent(
        build_agent_graph(RuleBasedChatModel()),
        [HumanMessage(content="vacation policy")],
    )
    assert "30 vacation days" in str(state["messages"][-1].content)
