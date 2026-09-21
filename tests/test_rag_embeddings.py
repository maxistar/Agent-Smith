import math

import pytest

from agentsmith.models import ModelConfigurationError
from agentsmith.rag import EducationalEmbeddings, cosine_similarity, create_embeddings


def test_educational_embeddings_are_stable_normalized_and_semantic() -> None:
    embeddings = EducationalEmbeddings()
    query = embeddings.embed_query("How much annual holiday do I get?")
    repeated = embeddings.embed_query("How much annual holiday do I get?")
    vacation = embeddings.embed_query("Employees receive vacation days.")
    expense = embeddings.embed_query("Submit a meal receipt.")

    assert query == repeated
    assert len(query) == embeddings.dimension == len(embeddings.concept_labels)
    assert math.isclose(math.sqrt(sum(value * value for value in query)), 1.0)
    assert cosine_similarity(query, vacation) > cosine_similarity(query, expense)


def test_zero_concept_embedding_is_stable_and_finite() -> None:
    vector = EducationalEmbeddings().embed_query("musical instruments")
    assert vector == [0.0] * len(vector)


def test_live_embeddings_require_credentials_without_exposing_values(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelConfigurationError) as captured:
        create_embeddings(live=True)
    assert "OPENAI_API_KEY" in str(captured.value)
    assert "replace-me" not in str(captured.value)
