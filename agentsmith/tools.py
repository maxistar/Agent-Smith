"""The stable example domain shared by every tool lesson."""

from langchain.tools import tool

POLICIES = {
    "vacation": "Employees have 30 vacation days per year.",
    "remote": "Employees may work remotely up to 3 days per week.",
    "parental": "Employees receive 16 weeks of paid parental leave.",
}
NOT_FOUND = "No matching policy was found."


@tool
def search_docs(query: str) -> str:
    """Search the fixed example company policies for a topic."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    normalized_query = query.casefold()
    for topic, policy in POLICIES.items():
        if topic in normalized_query:
            return policy
    return NOT_FOUND
