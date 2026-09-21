"""Lesson 20: score retrieval separately from grounding, citations, and tool use."""

from pathlib import Path

from agentsmith.rag import build_in_memory_store
from agentsmith.rag_evaluation import evaluate_rag_cases, load_rag_cases


def main() -> None:
    cases = load_rag_cases(Path(__file__).parents[1] / "data" / "rag_evaluation_cases.json")
    results = evaluate_rag_cases(build_in_memory_store(), cases)
    print("case                         retrieval grounding citations abstain tool overall")
    for result in results:
        print(
            f"{result.case_id:28} "
            f"{str(result.retrieval_hit):9} {str(result.grounded):9} "
            f"{str(result.citations_valid):9} {str(result.abstention_correct):7} "
            f"{str(result.tool_use_correct):4} {str(result.passed):7}"
        )


if __name__ == "__main__":
    main()
