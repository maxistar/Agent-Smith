## Context

AgentSmith currently provides ten offline-first lessons that progress from model messages and a deterministic dictionary-backed `search_docs` tool to LangGraph state, persistence, tracing, reliability, and evaluation. The fixed dictionary is intentionally useful for teaching tool calling, but it cannot demonstrate ingestion, chunking, semantic retrieval, citations, or the separation between retrieval quality and generation quality.

The new module must extend rather than replace that foundation. It introduces Chroma as a local vector store and RAG as a larger system containing both an indexing pipeline and a query-time pipeline. Learners must be able to run the complete module without external credentials, while explicitly opting into real OpenAI embeddings and generation when desired. Dependency resolution for `langchain-chroma` and `langchain-text-splitters` has been checked on Python 3.14, though Chroma adds a substantially heavier transitive dependency set than the current project.

## Goals / Non-Goals

**Goals:**

- Teach the ingestion path (`source → Document → chunks → embeddings → Chroma`) separately from query-time RAG (`question → retrieval → context → answer`).
- Add lessons 11–20 with one primary concept per lesson and retain independent execution from the repository root.
- Make embeddings and similarity observable through a transparent deterministic offline implementation before using Chroma abstractions.
- Demonstrate in-memory and persisted local Chroma collections, lifecycle operations, metadata filters, and stable IDs.
- Compare deterministic two-step RAG, model-selected agentic retrieval, and explicit LangGraph RAG routing.
- Require grounded answers with source attribution and correct abstention when retrieved evidence is insufficient.
- Evaluate retrieval and answer generation separately with a repeatable offline dataset.
- Preserve every lesson and test from the original learning path.

**Non-Goals:**

- Building a production document-ingestion service, crawler, PDF/OCR pipeline, web API, or user interface.
- Using Chroma Cloud, a separately managed Chroma server, distributed indexing, or multi-tenant authorization.
- Claiming that the deterministic educational embedding model represents production semantic quality.
- Teaching hybrid search, cross-encoder reranking, query rewriting, multimodal retrieval, or production prompt-injection defenses in this module.
- Persisting real or sensitive company documents in the repository or in test indexes.

## Decisions

### Continue the curriculum with lessons 11–20

The RAG module follows the existing numerical sequence:

11. construct LangChain `Document` objects from the synthetic policy corpus and inspect metadata;
12. split documents and compare chunk size, overlap, boundaries, and inherited metadata;
13. inspect document/query embeddings and similarity using a transparent offline embedding model;
14. index chunks in an in-memory Chroma collection and perform similarity search;
15. tune retrieval with `k`, scores, metadata filters, and maximum marginal relevance;
16. persist and reopen a Chroma index and demonstrate stable IDs plus add/update/upsert/delete behavior;
17. implement two-step RAG that always retrieves before generating a cited or abstaining answer;
18. expose retrieval as a tool so the model decides whether it is needed;
19. express RAG as an explicit LangGraph workflow with evidence grading and generate/abstain branches;
20. evaluate retrieval hit@k independently from answer grounding, citations, abstention, and expected tool use.

This ordering first teaches the data plane, then retrieval, then generation and control flow. Starting with a prebuilt RAG agent was rejected because it would hide the same model/application boundary that lessons 03–05 made explicit.

### Keep the original dictionary tool as a baseline

Lessons 01–10 and `search_docs` remain unchanged. The RAG module adds a distinct `retrieve_docs` capability backed by Chroma. Documentation will compare literal dictionary matching with semantic top-k retrieval rather than retrofitting Chroma into the earlier examples.

This preserves backward compatibility and makes the motivation for embeddings observable.

### Use a structured synthetic Markdown corpus

Add several non-sensitive policy documents under `knowledge/`, grouped by domain such as HR, security, and finance. Documents will contain multiple headings and enough text to cross chunk boundaries. Metadata will be derived deterministically from repository paths and an explicit small manifest or loader mapping, including at least `source`, `department`, and `policy_type`.

Questions will use paraphrases and include unanswerable cases. A three-line corpus was rejected because it cannot teach chunking, metadata filtering, or meaningful retrieval errors.

### Use an explicit educational embedding model offline

Implement a small LangChain `Embeddings`-compatible model whose dimensions correspond to documented concept groups and synonyms. It will normalize vectors and return stable output for both `embed_documents` and `embed_query`. Lesson 13 will print vectors and similarity calculations so the mechanism is inspectable.

The model is deliberately pedagogical, not production-grade. A fake random embedding was rejected because it cannot demonstrate meaningful ranking. A local transformer was rejected as the default because model downloads, native runtime cost, and opaque high-dimensional output conflict with the project's deterministic offline-first behavior.

When `--live` is supplied, use `OpenAIEmbeddings` configured by `AGENTSMITH_EMBEDDING_MODEL`, independently of the existing chat-model setting. Missing credentials must produce secret-safe guidance. All live embedding and generation calls remain opt-in.

### Use the LangChain Chroma integration over a local collection

Use the public `langchain_chroma.Chroma` vector-store interface so lessons operate on the same `Document` and retriever abstractions used elsewhere in LangChain. Start in memory, then introduce `persist_directory` in lesson 16. Chroma Cloud and server modes remain outside scope.

Collections used by tests must be isolated and temporary. Lesson collections must either be reset deliberately or use stable names and deterministic IDs so repeated runs do not silently duplicate data.

### Make indexing deterministic and rebuildable

Chunk IDs will derive from source identity and chunk position rather than random UUIDs. The standard lesson setup will support rebuilding a collection from the repository corpus. Persistent index directories are generated artifacts, excluded from Git, and safe to remove and recreate.

Incremental synchronization of arbitrary external corpora is out of scope. Lesson 16 will still demonstrate the semantic difference between add, update, upsert, and delete using a controlled record.

### Treat retrieval results as evidence with provenance

The retrieval layer returns structured documents and scores internally. Any tool-facing representation will include source metadata and bounded chunk text. Prompt construction will clearly delimit retrieved content from instructions and ask the generator to use only that evidence.

Answers must list the sources actually supplied to generation. If retrieval does not meet the configured evidence threshold, the two-step and graph workflows return a documented abstention instead of inventing an answer. Thresholds are configuration, not universal Chroma constants; lesson output will expose scores rather than implying they are interchangeable across embedding models.

### Separate three RAG control-flow styles

Two-step RAG always retrieves and is the simplest deterministic baseline. Agentic RAG exposes `retrieve_docs` as a tool and lets the model decide whether to call it. LangGraph RAG uses explicit nodes for retrieval, evidence grading, generation, and abstention.

These implementations may share loaders, embeddings, index construction, formatting, and evaluation helpers, but each lesson keeps its orchestration visible. This avoids repeating infrastructure without hiding the concept being taught.

### Evaluate retrieval before generation

The RAG dataset records the expected source set, whether retrieval/tool use is expected, answer fragments where deterministic, and whether the system should abstain. Evaluation reports at least:

- retrieval hit@k against expected sources;
- answer grounding against supplied context;
- presence and validity of cited sources;
- correct abstention for unsupported questions;
- expected retrieval-tool behavior for agentic cases.

Default evaluation uses offline embeddings and generation. Live evaluation is optional, reports nondeterministic results separately, and is not required for the offline test suite.

### Extend documentation without overloading the entry point

The README lesson table will include lessons 11–20 and a concise RAG quick start. A dedicated `docs/rag-overview.md` will explain indexing versus querying, Chroma's role, embedding consistency, persistence, control-flow variants, evaluation, privacy, and failure modes. The existing system overview will link to the new deeper document rather than duplicating it.

## Risks / Trade-offs

- **[Chroma adds many transitive dependencies]** → Keep it isolated to the RAG module, lock versions reproducibly, and validate installation and tests on Python 3.14 before building lessons.
- **[Educational embeddings can be mistaken for real semantic models]** → Name and document them explicitly as pedagogical, print their concept dimensions, and keep live embeddings behind `--live`.
- **[Chunking examples become artificial]** → Use multi-section documents with facts near boundaries and show both helpful and harmful split configurations.
- **[Repeated runs create duplicate or stale records]** → Use deterministic IDs, explicit collection reset/rebuild helpers, temporary test directories, and a documented lifecycle.
- **[Embedding model changes invalidate an existing collection]** → Store/document embedding identity with the index and rebuild rather than querying incompatible dimensions.
- **[Similarity scores are misread as universal confidence]** → Show ranking and score direction explicitly, use model-specific configurable thresholds, and avoid hard-coding a production claim.
- **[Retrieved content can influence instructions]** → Delimit context, instruct generation to treat it as evidence rather than commands, use only synthetic content, and document prompt injection as a later advanced topic.
- **[Persistent test data leaks into the repository]** → Put indexes under an ignored path and require tests to use disposable temporary directories.
- **[Agentic RAG is nondeterministic]** → Provide deterministic offline model doubles and evaluate expected tool use separately from final answer text.

## Migration Plan

1. Add and lock the Chroma and text-splitting dependencies while preserving Python 3.14.
2. Add the synthetic corpus, deterministic embeddings, loaders, index utilities, and offline tests.
3. Add lessons sequentially from document construction through evaluation.
4. Extend documentation and run all original and new offline lessons/tests.
5. If dependency or backward-compatibility checks fail, remove the new module and dependencies; lessons 01–10 require no data migration.

## Open Questions

- Live OpenAI embedding and generation smoke checks remain optional and depend on user-provided credentials, matching the policy established by the first learning-path change.
