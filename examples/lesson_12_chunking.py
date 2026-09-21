"""Lesson 12: compare chunk size, overlap, boundaries, and inherited metadata."""

from agentsmith.rag import load_policy_documents, split_documents


def describe(label: str, *, chunk_size: int, chunk_overlap: int) -> None:
    chunks = split_documents(
        load_policy_documents(), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print(f"\n{label}: size={chunk_size}, overlap={chunk_overlap}, chunks={len(chunks)}")
    for chunk in chunks[:4]:
        preview = chunk.page_content.replace("\n", " ")[:90]
        print(f"  {chunk.metadata} -> {preview!r}")


def main() -> None:
    print("Prerequisite: lesson 11. Observe how boundaries change while metadata survives.")
    describe("Focused", chunk_size=260, chunk_overlap=40)
    describe("Broad", chunk_size=520, chunk_overlap=80)


if __name__ == "__main__":
    main()
