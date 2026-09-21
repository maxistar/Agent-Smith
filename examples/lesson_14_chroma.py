"""Lesson 14: index policy chunks in memory and run semantic Chroma search."""

from agentsmith.rag import build_in_memory_store, retrieve
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        embeddings = configured_embeddings(live=args.live)
        store = build_in_memory_store(embeddings=embeddings)
        print("Lower cosine distance means closer for this collection; it is not probability.")
        for rank, item in enumerate(
            retrieve(store, "How much annual holiday do employees get?", k=3), start=1
        ):
            preview = item.document.page_content.replace("\n", " ")[:120]
            print(f"{rank}. distance={item.distance:.3f} source={item.source} {preview!r}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
