## 1. Dependencies and configuration

- [x] 1.1 Add `langchain-chroma` and `langchain-text-splitters` to `pyproject.toml`, refresh `uv.lock`, and verify the resolved environment supports Python 3.14.
- [x] 1.2 Add a Git-ignored default directory for generated local Chroma indexes without changing the existing `.env` and credential exclusions.
- [x] 1.3 Extend shared CLI/configuration helpers with an explicit `--live` embedding mode and `AGENTSMITH_EMBEDDING_MODEL`, including secret-safe errors for missing provider configuration.

## 2. Synthetic corpus and document loading

- [x] 2.1 Add several non-sensitive Markdown policy documents under `knowledge/` for HR, security, and finance, with multiple sections and facts that exercise chunk boundaries.
- [x] 2.2 Implement deterministic corpus loading into LangChain `Document` objects with stable `source`, `department`, and `policy_type` metadata.
- [x] 2.3 Add lesson 11 to inspect document content and provenance, and document its prerequisite and expected observations in the module output.
- [x] 2.4 Add offline tests for corpus completeness, stable ordering, required metadata, and absence of credential or network requirements.

## 3. Chunking

- [x] 3.1 Implement shared text-splitting helpers that preserve provenance and attach stable source-relative chunk positions.
- [x] 3.2 Add lesson 12 to compare two chunk-size/overlap configurations and display chunk counts, boundaries, and inherited metadata.
- [x] 3.3 Add tests proving chunking is deterministic, preserves all required metadata, and visibly changes under the compared configurations.

## 4. Educational and live embeddings

- [x] 4.1 Implement a normalized, deterministic LangChain `Embeddings` implementation with documented concept and synonym dimensions for the synthetic corpus.
- [x] 4.2 Implement the opt-in OpenAI embeddings factory using `AGENTSMITH_EMBEDDING_MODEL` while keeping the default path fully offline.
- [x] 4.3 Add lesson 13 to print vectors and similarity scores for paraphrased and unrelated examples and clearly label the offline embedding as educational rather than production-grade.
- [x] 4.4 Add tests for stable vector dimensions, normalization, document/query consistency, meaningful paraphrase ranking, and live-mode configuration errors without exposing secrets.

## 5. In-memory Chroma and retrieval controls

- [x] 5.1 Implement deterministic chunk IDs and an isolated in-memory Chroma collection builder using the LangChain Chroma integration.
- [x] 5.2 Add lesson 14 to index the corpus and display ranked similarity-search results with content, source metadata, and score direction explained.
- [x] 5.3 Implement shared retrieval options for `k`, metadata filters, similarity-with-score, and maximum marginal relevance without treating raw scores as universal probabilities.
- [x] 5.4 Add lesson 15 to compare similarity and MMR results and demonstrate a department metadata filter over the same collection.
- [x] 5.5 Add offline tests for stable IDs, collection counts, paraphrase retrieval, bounded top-k results, metadata filters, and similarity-versus-MMR behavior.

## 6. Persistent Chroma lifecycle

- [x] 6.1 Implement a rebuildable persistent-index helper with a deterministic collection name, explicit reset behavior, and recorded embedding identity/dimension.
- [x] 6.2 Add lesson 16 to build, close, reopen, and query a local index and to demonstrate add, update, upsert, and delete on controlled records.
- [x] 6.3 Detect incompatible persisted embedding dimensions and return actionable rebuild guidance instead of continuing with misleading results.
- [x] 6.4 Add tests using temporary directories for reopen-without-reingestion, repeatable rebuilds without duplicates, CRUD behavior, and dimension-mismatch handling.

## 7. Grounded two-step RAG

- [x] 7.1 Implement retrieval-result formatting that bounds chunk text, preserves source attribution, and clearly delimits evidence from instructions.
- [x] 7.2 Implement the two-step RAG path that always retrieves, applies a configurable evidence threshold, and either generates from supplied evidence or returns the documented abstention.
- [x] 7.3 Add deterministic offline generation plus opt-in live generation while requiring citations to refer only to sources supplied in the retrieval context.
- [x] 7.4 Add lesson 17 with one supported and one unsupported question so learners can inspect grounded citation and abstention behavior.
- [x] 7.5 Add tests for retrieval-before-generation, valid citations, threshold handling, unsupported-question abstention, and resistance to inventing sources.

## 8. Agent-selected retrieval

- [x] 8.1 Implement a bounded `retrieve_docs` tool backed by Chroma that returns source-attributed evidence without changing the existing `search_docs` tool.
- [x] 8.2 Add deterministic offline model behavior for retrieval and no-retrieval cases and bind the same tool contract to the live chat model.
- [x] 8.3 Add lesson 18 to show an internal-policy question that invokes `retrieve_docs` and a conversational question that bypasses retrieval.
- [x] 8.4 Add tests for tool-call selection, bounded result size, tool-call/result correlation, grounded final output, and the no-tool route.

## 9. Explicit LangGraph RAG workflow

- [x] 9.1 Define typed graph state containing the question, retrieved documents, scores or evidence grade, answer, and cited sources.
- [x] 9.2 Implement visible retrieve, grade, generate, and abstain nodes with conditional routing based on evidence sufficiency.
- [x] 9.3 Add lesson 19 to print or otherwise expose both `retrieve → grade → generate → end` and `retrieve → grade → abstain → end` routes.
- [x] 9.4 Add tests for both graph branches, state propagation, cited-source integrity, and the guarantee that insufficient evidence does not invoke generation.

## 10. Retrieval-aware evaluation

- [x] 10.1 Add an offline RAG evaluation dataset containing answerable, paraphrased, metadata-filtered, conversational no-retrieval, and unsupported cases with expected sources and behavior.
- [x] 10.2 Implement separate metrics for retrieval hit@k, answer grounding, citation validity, correct abstention, and expected agentic tool use.
- [x] 10.3 Add lesson 20 to report per-case and aggregate results while keeping retrieval failures distinct from plausible answer text.
- [x] 10.4 Add tests for every evaluation category, including a plausible answer with the wrong retrieved source and an unsupported case with an unrelated citation.

## 11. Learner documentation

- [x] 11.1 Extend the README lesson table and quick start with lessons 11–20, offline and live commands, embedding configuration, persistent-index creation, and cleanup.
- [x] 11.2 Add `docs/rag-overview.md` with indexing/query-flow diagrams, Chroma's role, embedding consistency, retrieval controls, the three RAG control-flow styles, evaluation, privacy, and common failure modes.
- [x] 11.3 Link the new RAG overview from `docs/system-overview.md` and compare `search_docs` literal lookup with the new semantic `retrieve_docs` tool without rewriting lessons 01–10.

## 12. End-to-end verification

- [x] 12.1 Run the complete offline test suite without OpenAI, LangSmith, or Chroma Cloud credentials and verify all index writes remain inside test temporary directories.
- [x] 12.2 Run lessons 11–20 from the repository root in offline mode and verify each lesson works independently after documented setup.
- [x] 12.3 Re-run lessons 01–10 and their tests to confirm their commands and observable behavior remain unchanged.
- [x] 12.4 Run `uv run ruff check .`, `uv run ruff format --check .`, and a clean-install dependency check on Python 3.14.
- [x] 12.5 Verify the documented live configuration path without making billable calls; record real OpenAI embedding/generation smoke tests as consciously excluded when credentials are unavailable.
