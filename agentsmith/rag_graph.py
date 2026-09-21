"""Explicit retrieve-grade-generate LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph

from agentsmith.rag import (
    ABSTENTION,
    EducationalEmbeddings,
    RetrievedChunk,
    answer_from_results,
    retrieve,
)


class RagState(TypedDict, total=False):
    question: str
    documents: list[Document]
    distances: list[float]
    evidence_sufficient: bool
    answer: str
    sources: list[str]
    route: list[str]


def build_rag_graph(
    vector_store: Any,
    *,
    embeddings: Embeddings | None = None,
    max_distance: float = 0.55,
    live: bool = False,
):
    selected = embeddings or EducationalEmbeddings()

    def retrieve_node(state: RagState) -> RagState:
        results = retrieve(vector_store, state["question"], k=3)
        return {
            "documents": [result.document for result in results],
            "distances": [result.distance for result in results],
            "route": [*state.get("route", []), "retrieve"],
        }

    def grade_node(state: RagState) -> RagState:
        sufficient = bool(state["distances"] and min(state["distances"]) <= max_distance)
        return {
            "evidence_sufficient": sufficient,
            "route": [*state.get("route", []), "grade"],
        }

    def generate_node(state: RagState) -> RagState:
        results = [
            RetrievedChunk(document=document, distance=distance)
            for document, distance in zip(state["documents"], state["distances"], strict=True)
        ]
        result = answer_from_results(
            state["question"],
            results,
            embeddings=selected,
            live=live,
            max_distance=max_distance,
        )
        return {
            "answer": result.answer,
            "sources": list(result.sources),
            "route": [*state.get("route", []), "generate"],
        }

    def abstain_node(state: RagState) -> RagState:
        return {
            "answer": ABSTENTION,
            "sources": [],
            "route": [*state.get("route", []), "abstain"],
        }

    def route_after_grade(state: RagState) -> str:
        return "generate" if state["evidence_sufficient"] else "abstain"

    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("abstain", abstain_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", route_after_grade, {"generate": "generate", "abstain": "abstain"}
    )
    builder.add_edge("generate", END)
    builder.add_edge("abstain", END)
    return builder.compile()
