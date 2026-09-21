## Why

The existing learning path explains tools and agent orchestration using a fixed dictionary, but it stops before showing how applications retrieve relevant context from a larger document corpus. Adding an offline-first RAG and Chroma module will let learners see the complete indexing, retrieval, grounded-generation, and evaluation lifecycle without obscuring the concepts behind a prebuilt agent.

## What Changes

- Extend the numbered curriculum with lessons 11–20 covering LangChain documents, chunking, embeddings, Chroma collections, retrieval tuning, persistent indexing, two-step RAG, agentic RAG, LangGraph RAG control flow, and RAG evaluation.
- Add a multi-document synthetic policy corpus with stable source metadata and enough structure to demonstrate meaningful chunking, filtering, citations, and correct abstention.
- Preserve offline-first execution with transparent deterministic educational embeddings and scripted generation; allow explicit `--live` runs to use the configured OpenAI chat and embedding models.
- Add local in-memory and persistent Chroma examples with stable document IDs, metadata filters, update/upsert/delete behavior, index reopening, and dimension-consistency guidance.
- Keep the original `search_docs` dictionary tool and lessons 01–10 unchanged so learners can compare literal lookup with vector retrieval.
- Add offline tests and an evaluation dataset that measure retrieval source hit rate separately from answer grounding, citation behavior, tool use, and abstention.
- Extend the README and system overview with the RAG indexing/query split, lesson commands, local-index lifecycle, privacy considerations, and troubleshooting.

## Capabilities

### New Capabilities

- `rag-chroma-learning-path`: A progressive, independently runnable curriculum for document ingestion, embeddings, Chroma retrieval, grounded RAG workflows, agentic retrieval, and retrieval-aware evaluation.

### Modified Capabilities

None.

## Impact

- Adds `langchain-chroma`, `langchain-text-splitters`, ChromaDB, and their transitive runtime dependencies; dependency resolution has been checked against the project's Python 3.14 baseline.
- Adds synthetic source documents, chunk/index utilities, embedding configuration, retrieval tools, RAG graph components, evaluation cases, and tests.
- Adds a Git-ignored local Chroma persistence directory that learners can safely rebuild from the synthetic corpus.
- Extends learner documentation and the numbered examples while retaining backward compatibility with lessons 01–10 and their existing offline behavior.
- Live embeddings and generation may send document chunks and questions to the configured provider and may incur cost; they remain explicit opt-in behavior.
