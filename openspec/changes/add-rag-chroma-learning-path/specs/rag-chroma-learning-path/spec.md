## ADDED Requirements

### Requirement: Ordered RAG learning module
The project SHALL provide lessons 11–20 as an ordered extension of the existing learning path, progressing through documents, chunking, embeddings, Chroma indexing, retrieval tuning, persistence, two-step RAG, agentic RAG, LangGraph RAG, and evaluation. Each lesson SHALL introduce one primary concept and identify its prerequisites and expected observations.

#### Scenario: Learner enters the RAG module
- **WHEN** a learner completes or reviews lesson 10 and opens the project lesson map
- **THEN** lessons 11–20 appear in one unambiguous order that separates indexing concepts from query-time RAG concepts

### Requirement: Independently runnable RAG lessons
Each RAG lesson SHALL run directly from the repository root after documented setup without modifying source code or requiring later lessons. Default execution SHALL use local synthetic data, deterministic embeddings, and offline model behavior without provider or Chroma Cloud credentials.

#### Scenario: Run RAG lesson offline
- **WHEN** a learner runs any lesson 11–20 without OpenAI, LangSmith, or Chroma Cloud credentials
- **THEN** the lesson executes its primary educational path without attempting an external network request

### Requirement: Synthetic source corpus with provenance
The project SHALL provide a non-sensitive, multi-document policy corpus containing multiple sections and enough content to demonstrate chunk boundaries. Every loaded document and derived chunk SHALL carry stable provenance metadata including source, department, and policy type.

#### Scenario: Load source documents
- **WHEN** lesson 11 loads the repository corpus
- **THEN** it displays each `Document` with its content source and required metadata without exposing sensitive data

### Requirement: Observable chunking behavior
The learning path SHALL demonstrate how chunk size, overlap, and document boundaries affect the chunks supplied to indexing. Chunking SHALL preserve source metadata and stable source-relative chunk positions.

#### Scenario: Compare split configurations
- **WHEN** lesson 12 splits the same source with two documented configurations
- **THEN** it displays different chunk boundaries and counts while preserving the originating source metadata on every chunk

### Requirement: Deterministic educational embeddings
The default embedding implementation SHALL satisfy the LangChain embeddings interface, produce stable normalized vectors for documents and queries, and expose documented concept dimensions or synonym groups that make similarity rankings explainable. Documentation SHALL explicitly state that it is educational and not suitable for production semantic search.

#### Scenario: Inspect semantic similarity offline
- **WHEN** lesson 13 embeds a paraphrased query and several policy chunks
- **THEN** it displays stable vectors or similarity scores and ranks the conceptually matching chunk ahead of unrelated chunks without a network call

#### Scenario: Select live embeddings
- **WHEN** a learner supplies `--live` with valid provider configuration
- **THEN** the lesson uses the configured live embedding model and clearly identifies the networked, potentially billable mode

### Requirement: In-memory Chroma indexing and search
The learning path SHALL demonstrate adding stable-ID chunks with metadata to an in-memory Chroma collection and querying them through the LangChain Chroma integration. Search output SHALL expose retrieved content, source metadata, rank, and score where available.

#### Scenario: Retrieve a paraphrased policy question
- **WHEN** lesson 14 indexes the synthetic chunks and searches using a documented paraphrase
- **THEN** the relevant policy chunk is returned ahead of unrelated chunks with its source metadata

### Requirement: Retrieval controls
The project SHALL demonstrate the observable effects of top-k selection, similarity scores, metadata filtering, and maximum marginal relevance. It SHALL not present raw similarity or distance values as universal confidence probabilities.

#### Scenario: Filter by metadata
- **WHEN** lesson 15 applies a department filter to a query spanning multiple policy domains
- **THEN** all returned chunks satisfy the requested department metadata constraint

#### Scenario: Compare retrieval strategies
- **WHEN** lesson 15 runs similarity and maximum-marginal-relevance retrieval over the same collection
- **THEN** it displays the selected sources so the learner can compare relevance and diversity

### Requirement: Rebuildable persistent Chroma index
The project SHALL support a local persistent Chroma index under a Git-ignored path, deterministic collection naming, stable chunk IDs, explicit rebuild behavior, and reopening the index in a later process or object instance. The persistence lesson SHALL distinguish add, update, upsert, and delete behavior.

#### Scenario: Reopen persisted index
- **WHEN** lesson 16 builds an index, creates a new Chroma instance for the same persistence path and embedding configuration, and repeats a query
- **THEN** the new instance retrieves the previously indexed document without re-ingesting the corpus

#### Scenario: Rebuild without duplicates
- **WHEN** the standard index builder runs repeatedly for the unchanged corpus
- **THEN** the resulting collection contains one record for each stable chunk ID rather than duplicate records

#### Scenario: Embedding configuration is incompatible
- **WHEN** a persisted collection is opened with an embedding configuration whose vector dimension is incompatible
- **THEN** the lesson reports that the index must be rebuilt instead of silently returning misleading results

### Requirement: Grounded two-step RAG
The two-step RAG lesson SHALL always retrieve before generation, clearly delimit retrieved evidence, instruct generation to use only that evidence, and return source citations corresponding to supplied chunks. When evidence does not satisfy the configured threshold, it SHALL return a documented abstention instead of generating an unsupported answer.

#### Scenario: Answer supported question with citation
- **WHEN** lesson 17 receives a question supported by the corpus
- **THEN** it returns an answer grounded in retrieved text and cites at least one retrieved source

#### Scenario: Abstain from unsupported answer
- **WHEN** lesson 17 receives a question for which retrieved evidence is insufficient
- **THEN** it returns the documented abstention and does not invent a source or policy

### Requirement: Agent-selected retrieval tool
The agentic RAG lesson SHALL expose Chroma retrieval as a bounded `retrieve_docs` tool and let the chat model decide whether to call it. Tool results SHALL include source-attributed evidence while limiting returned chunk count and content size.

#### Scenario: Agent retrieves internal policy
- **WHEN** lesson 18 receives an internal-policy question
- **THEN** the agent calls `retrieve_docs`, receives source-attributed chunks, and produces a grounded final response

#### Scenario: Agent answers without retrieval
- **WHEN** lesson 18 receives a simple conversational question that requires no corpus evidence
- **THEN** the agent produces a final response without calling `retrieve_docs`

### Requirement: Explicit LangGraph RAG workflow
The LangGraph RAG lesson SHALL implement visible nodes for retrieval, evidence grading, generation, and abstention, with conditional routing based on evidence sufficiency. Its state SHALL expose the question, retrieved documents, scores or grade, answer, and cited sources needed to inspect the route.

#### Scenario: Sufficient evidence route
- **WHEN** lesson 19 retrieves evidence that passes the configured grading rule
- **THEN** the graph follows `retrieve → grade → generate → end` and returns a cited answer

#### Scenario: Insufficient evidence route
- **WHEN** lesson 19 retrieves no evidence or evidence below the configured grading rule
- **THEN** the graph follows `retrieve → grade → abstain → end` without invoking unsupported generation

### Requirement: Retrieval-aware evaluation
The project SHALL provide an offline RAG evaluation dataset containing answerable, paraphrased, filtered, no-retrieval, and unsupported questions. Evaluation SHALL report retrieval hit@k against expected sources separately from answer grounding, citation validity, correct abstention, and expected tool-use behavior.

#### Scenario: Retrieval fails but generation appears plausible
- **WHEN** an evaluation case does not retrieve its expected source even if the final answer contains plausible text
- **THEN** lesson 20 reports retrieval failure independently from answer-level metrics

#### Scenario: Evaluate unsupported question
- **WHEN** an evaluation case is marked unsupported
- **THEN** lesson 20 records success only when the workflow abstains without citing an unrelated source

### Requirement: Backward-compatible offline verification
Adding the RAG module SHALL NOT change the observable behavior or documented commands of lessons 01–10. The default automated suite SHALL validate corpus loading, deterministic embeddings, chunk metadata, Chroma indexing, persistence, retrieval routes, citations, abstention, and evaluation without network credentials and using disposable index locations.

#### Scenario: Run complete offline suite
- **WHEN** tests run without OpenAI, LangSmith, or Chroma Cloud credentials
- **THEN** all original and RAG tests complete without external requests or writes outside test-controlled temporary directories

### Requirement: RAG learner documentation
The README SHALL include lessons 11–20, dependency/setup guidance, offline and live commands, and persistent-index cleanup instructions. A dedicated RAG overview SHALL explain indexing versus query-time flow, Chroma's role, embedding consistency, retrieval strategies, RAG control-flow variants, evaluation metrics, privacy implications, and common failure modes.

#### Scenario: Learner prepares the RAG module
- **WHEN** a learner follows the documented RAG setup on a clean checkout
- **THEN** the learner can install dependencies, run lesson 11 offline, build and remove a local persistent index, and run RAG tests using documented commands
