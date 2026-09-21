from agentsmith.rag import load_policy_documents, split_documents


def test_corpus_is_stable_synthetic_and_has_provenance() -> None:
    documents = load_policy_documents()
    assert [item.metadata["source"] for item in documents] == [
        "finance/expense-policy.md",
        "hr/leave-policy.md",
        "security/access-policy.md",
    ]
    assert all(len(item.page_content) > 300 for item in documents)
    assert all(
        {"source", "department", "policy_type"} <= item.metadata.keys() for item in documents
    )


def test_chunking_is_deterministic_preserves_metadata_and_changes_with_configuration() -> None:
    documents = load_policy_documents()
    focused = split_documents(documents, chunk_size=260, chunk_overlap=40)
    repeated = split_documents(documents, chunk_size=260, chunk_overlap=40)
    broad = split_documents(documents, chunk_size=520, chunk_overlap=80)

    assert focused == repeated
    assert len(focused) > len(broad)
    assert all(
        {"source", "department", "policy_type", "chunk_index"} <= item.metadata.keys()
        for item in focused
    )
    for source in {item.metadata["source"] for item in focused}:
        positions = [
            item.metadata["chunk_index"] for item in focused if item.metadata["source"] == source
        ]
        assert positions == list(range(len(positions)))
