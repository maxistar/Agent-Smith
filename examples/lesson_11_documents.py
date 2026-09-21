"""Lesson 11: load LangChain Documents and inspect stable provenance metadata."""

from agentsmith.rag import load_policy_documents


def main() -> None:
    print("Prerequisite: lessons 01-10. Observe content plus source provenance.")
    for document in load_policy_documents():
        print(f"\n{document.metadata}")
        print(document.page_content.splitlines()[0])


if __name__ == "__main__":
    main()
