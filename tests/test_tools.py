import pytest

from agentsmith.tools import NOT_FOUND, search_docs


def test_search_docs_finds_known_topic_case_insensitively() -> None:
    assert search_docs.invoke({"query": "Tell me about VACATION"}) == (
        "Employees have 30 vacation days per year."
    )


def test_search_docs_returns_explicit_not_found_result() -> None:
    assert search_docs.invoke({"query": "dress code"}) == NOT_FOUND


@pytest.mark.parametrize("query", ["", "   "])
def test_search_docs_rejects_empty_queries(query: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        search_docs.invoke({"query": query})


def test_search_docs_rejects_non_string_query() -> None:
    with pytest.raises(Exception, match="string"):
        search_docs.invoke({"query": None})
