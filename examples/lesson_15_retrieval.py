"""Lesson 15: compare k, scores, metadata filters, and MMR diversity."""

from agentsmith.rag import build_in_memory_store, retrieve, retrieve_mmr
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        store = build_in_memory_store(embeddings=configured_embeddings(live=args.live))
        query = "vacation security devices and expense receipts"
        similarity = retrieve(store, query, k=3)
        filtered = retrieve(store, query, k=3, metadata_filter={"department": "finance"})
        mmr = retrieve_mmr(store, query, k=3, fetch_k=8)
        print("Similarity:", [item.source for item in similarity])
        print("Finance filter:", [item.source for item in filtered])
        print("MMR diversity:", [item.metadata["source"] for item in mmr])
        print("Scores rank results for one embedding setup; they are not confidence probabilities.")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
