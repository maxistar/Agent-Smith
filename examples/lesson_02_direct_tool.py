"""Lesson 02: call a deterministic tool directly, without an LLM."""

from agentsmith.tools import search_docs


def main() -> None:
    query = "vacation"
    print(f"Tool input: {query!r}")
    print(f"Tool output: {search_docs.invoke({'query': query})}")


if __name__ == "__main__":
    main()
