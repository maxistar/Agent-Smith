"""Lesson 17: always retrieve, then answer with citations or abstain."""

from agentsmith.rag import answer_with_rag, build_in_memory_store
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        embeddings = configured_embeddings(live=args.live)
        store = build_in_memory_store(embeddings=embeddings)
        for question in (
            "How much annual holiday do employees get?",
            "What is the company policy on musical instruments?",
        ):
            result = answer_with_rag(question, store, embeddings=embeddings, live=args.live)
            print(f"\nQuestion: {question}\n{result.answer}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
