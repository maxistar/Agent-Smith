"""Lesson 18: let the model decide whether policy retrieval is necessary."""

from agentsmith.agent import format_trajectory
from agentsmith.rag import build_in_memory_store
from agentsmith.rag_agent import run_agentic_rag
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        store = build_in_memory_store(embeddings=configured_embeddings(live=args.live))
        for question in ("What is the lost-device policy?", "Hello, how are you?"):
            messages, used = run_agentic_rag(store, question, live=args.live)
            print(f"\nQuestion: {question}\nretrieve_docs used: {used}")
            print(format_trajectory(messages))

    run_lesson(lesson)


if __name__ == "__main__":
    main()
