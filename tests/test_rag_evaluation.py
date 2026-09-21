from pathlib import Path

from agentsmith.rag import build_in_memory_store
from agentsmith.rag_evaluation import evaluate_rag_cases, load_rag_cases, score_rag_case


def test_rag_evaluation_covers_all_categories_and_passes_offline() -> None:
    cases = load_rag_cases(Path("data/rag_evaluation_cases.json"))
    assert {case["id"] for case in cases} == {
        "answerable-vacation",
        "paraphrased-lost-device",
        "filtered-expense-domain",
        "no-retrieval-greeting",
        "unsupported-space-policy",
    }
    results = evaluate_rag_cases(build_in_memory_store(), cases)
    assert all(result.passed for result in results)
    filtered = next(case for case in cases if case["id"] == "filtered-expense-domain")
    assert filtered["metadata_filter"] == {"department": "finance"}


def test_plausible_answer_does_not_hide_retrieval_failure() -> None:
    case = {
        "id": "wrong-source",
        "expected_sources": ["hr/leave-policy.md"],
        "expected_contains": "30 paid vacation days",
        "expected_tool_use": True,
        "unsupported": False,
    }
    result = score_rag_case(
        case,
        answer="Employees receive 30 paid vacation days.",
        cited_sources=["finance/expense-policy.md"],
        retrieved_sources=["finance/expense-policy.md"],
        tool_used=True,
    )
    assert result.grounded
    assert not result.retrieval_hit


def test_unsupported_case_fails_when_it_cites_unrelated_source() -> None:
    case = {
        "id": "unsupported",
        "expected_sources": [],
        "expected_contains": "not enough evidence",
        "expected_tool_use": True,
        "unsupported": True,
    }
    result = score_rag_case(
        case,
        answer="A confident invented answer.",
        cited_sources=["finance/expense-policy.md"],
        retrieved_sources=["finance/expense-policy.md"],
        tool_used=True,
    )
    assert not result.abstention_correct
    assert not result.grounded
