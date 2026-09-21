from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentsmith.rag import (
    EducationalEmbeddings,
    build_in_memory_store,
    build_persistent_store,
    load_policy_documents,
    reopen_persistent_store,
    retrieve,
    retrieve_mmr,
    split_documents,
    stable_chunk_id,
)


class IncompatibleEmbeddings(Embeddings):
    identity = "incompatible-v2"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_in_memory_chroma_uses_stable_ids_scores_and_semantic_ranking() -> None:
    chunks = split_documents(load_policy_documents())
    assert [stable_chunk_id(item) for item in chunks] == [stable_chunk_id(item) for item in chunks]

    store = build_in_memory_store()
    results = retrieve(store, "How much annual holiday do employees get?", k=3)
    assert len(results) == 3
    assert results[0].source == "hr/leave-policy.md"
    assert results[0].distance <= results[1].distance


def test_filters_and_mmr_retrieval_controls() -> None:
    store = build_in_memory_store()
    query = "vacation security devices and expense receipts"
    filtered = retrieve(store, query, k=3, metadata_filter={"department": "finance"})
    diversified = retrieve_mmr(store, query, k=3, fetch_k=8)

    assert filtered
    assert all(item.document.metadata["department"] == "finance" for item in filtered)
    assert len({item.metadata["source"] for item in diversified}) >= 2


def test_persistent_index_reopens_rebuilds_without_duplicates_and_checks_dimensions(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "chroma"
    embeddings = EducationalEmbeddings()
    first = build_persistent_store(directory, embeddings=embeddings, rebuild=True)
    first_count = first._collection.count()
    second = build_persistent_store(directory, embeddings=embeddings)
    assert second._collection.count() == first_count

    reopened = reopen_persistent_store(directory, embeddings=embeddings)
    assert retrieve(reopened, "stolen laptop", k=1)[0].source == "security/access-policy.md"

    with pytest.raises(ValueError, match="rebuild"):
        reopen_persistent_store(directory, embeddings=IncompatibleEmbeddings())


def test_chroma_lifecycle_add_update_upsert_and_delete(tmp_path: Path) -> None:
    store = build_persistent_store(tmp_path / "lifecycle", rebuild=True)
    document = Document(
        page_content="Temporary travel policy.",
        metadata={
            "source": "demo/lifecycle.md",
            "department": "finance",
            "policy_type": "demo",
            "chunk_index": 0,
        },
    )
    record_id = stable_chunk_id(document)
    store.add_documents([document], ids=[record_id])
    assert store.get(ids=[record_id])["documents"] == ["Temporary travel policy."]

    updated = Document(page_content="Updated travel policy.", metadata=document.metadata)
    store.update_documents(ids=[record_id], documents=[updated])
    assert store.get(ids=[record_id])["documents"] == ["Updated travel policy."]

    upserted = Document(page_content="Upserted travel policy.", metadata=document.metadata)
    store.add_documents([upserted], ids=[record_id])
    assert store.get(ids=[record_id])["documents"] == ["Upserted travel policy."]

    store.delete(ids=[record_id])
    assert store.get(ids=[record_id])["ids"] == []
