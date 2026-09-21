"""Lesson 16: rebuild, mutate, and reopen a persistent local Chroma index."""

from langchain_core.documents import Document

from agentsmith.rag import (
    DEFAULT_INDEX_DIR,
    build_persistent_store,
    reopen_persistent_store,
    retrieve,
    stable_chunk_id,
)
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    parser = lesson_parser(__doc__)
    parser.add_argument(
        "--rebuild", action="store_true", help="Delete and rebuild the lesson index."
    )
    args = parser.parse_args()

    def lesson() -> None:
        embeddings = configured_embeddings(live=args.live)
        store = build_persistent_store(
            DEFAULT_INDEX_DIR, embeddings=embeddings, rebuild=args.rebuild
        )
        demo = Document(
            page_content="Temporary demo travel policy.",
            metadata={
                "source": "demo/lifecycle.md",
                "department": "finance",
                "policy_type": "demo",
                "chunk_index": 0,
            },
        )
        demo_id = stable_chunk_id(demo)
        store.add_documents([demo], ids=[demo_id])
        updated = Document(page_content="Updated temporary travel policy.", metadata=demo.metadata)
        store.update_documents(ids=[demo_id], documents=[updated])
        store.add_documents([updated], ids=[demo_id])  # Chroma upsert via the integration
        store.delete(ids=[demo_id])
        reopened = reopen_persistent_store(DEFAULT_INDEX_DIR, embeddings=embeddings)
        result = retrieve(reopened, "How quickly must a stolen laptop be reported?", k=1)[0]
        print(f"Reopened result: {result.source} distance={result.distance:.3f}")
        print(f"Persistent index: {DEFAULT_INDEX_DIR}")
        print("Stable IDs make rebuilds repeatable; remove this generated directory to clean up.")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
