"""Lesson 13: inspect explainable educational embeddings and semantic similarity."""

from agentsmith.rag import EducationalEmbeddings, cosine_similarity
from examples._shared import configured_embeddings, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        embeddings = configured_embeddings(live=args.live)
        examples = [
            "How much annual holiday do I get?",
            "Employees receive 30 vacation days.",
            "Receipts are required for large expenses.",
        ]
        vectors = embeddings.embed_documents(examples)
        if isinstance(embeddings, EducationalEmbeddings):
            print(f"Educational concept dimensions: {embeddings.concept_labels}")
            print("These transparent vectors are for teaching, not production search.")
        for text, vector in zip(examples, vectors, strict=True):
            print(f"{text!r}\n  vector={vector}")
        print(f"holiday↔vacation: {cosine_similarity(vectors[0], vectors[1]):.3f}")
        print(f"holiday↔expense:  {cosine_similarity(vectors[0], vectors[2]):.3f}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
