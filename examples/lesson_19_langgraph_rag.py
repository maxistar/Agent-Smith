"""Lesson 19: inspect explicit retrieve, grade, generate, and abstain graph routes."""

from agentsmith.rag import build_in_memory_store
from agentsmith.rag_graph import build_rag_graph
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        embeddings = configured_embeddings(live=args.live)
        store = build_in_memory_store(embeddings=embeddings)
        graph = build_rag_graph(store, embeddings=embeddings, live=args.live)
        for question in (
            "How much annual holiday do employees get?",
            "What is the company policy on musical instruments?",
        ):
            result = graph.invoke({"question": question, "route": []})
            print(f"\nQuestion: {question}")
            print("Route:", " -> ".join([*result["route"], "end"]))
            print(result["answer"])

    run_lesson(lesson)


if __name__ == "__main__":
    main()
