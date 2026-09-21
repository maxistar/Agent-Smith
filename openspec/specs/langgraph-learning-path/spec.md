# langgraph-learning-path Specification

## Purpose

Define a progressive, independently runnable curriculum that teaches how model messages, tools, LangGraph orchestration, state, persistence, observability, reliability, and evaluation fit together in an LLM application.

## Requirements

### Requirement: Ordered learning path
The project SHALL provide a numbered learning path that progresses from a basic chat-model invocation through tools, manual orchestration, LangGraph orchestration, state, persistence, tracing, reliability, and evaluation. Each lesson SHALL identify one primary new learning objective and build only on concepts introduced by earlier lessons.

#### Scenario: Learner follows the sequence
- **WHEN** a learner reads the project documentation from the beginning
- **THEN** the lessons are presented in a single unambiguous order with the objective and prerequisite of each lesson

### Requirement: Independently runnable lessons
Each lesson SHALL be runnable directly from the repository root after the documented setup, without editing the lesson source or completing later lessons. Each lesson SHALL expose enough input, intermediate state, and output to make its primary concept observable.

#### Scenario: Earlier lesson remains runnable
- **WHEN** the complete learning path is installed and a learner runs any earlier numbered lesson
- **THEN** that lesson executes independently of later lesson entry points and displays the behavior relevant to its objective

### Requirement: Explicit model message fundamentals
The first live-model lesson SHALL demonstrate a chat-model invocation and SHALL expose the returned message type and content before introducing tools or graph orchestration.

#### Scenario: Basic model invocation
- **WHEN** valid model configuration is present and the learner runs the first model lesson
- **THEN** the lesson sends a documented user message and displays the resulting assistant message in an inspectable form

### Requirement: Progressive tool-call explanation
The learning path SHALL separately demonstrate direct tool execution, model-generated tool requests, and application-side execution of those requests. It SHALL make clear that binding a tool gives the model its schema but does not cause the model itself to execute Python code.

#### Scenario: Inspect tool request without execution
- **WHEN** the configured model responds with a request for `search_docs`
- **THEN** the tool-request lesson displays the requested tool name, arguments, and call identifier without automatically executing the tool

#### Scenario: Execute tool request manually
- **WHEN** the manual-loop lesson receives a valid `search_docs` tool request
- **THEN** the application invokes the matching tool, appends a tool-result message associated with the original call identifier, and invokes the model again for a final response

### Requirement: Deterministic shared knowledge tool
The project SHALL provide a deterministic `search_docs` tool with a documented input contract, fixed non-sensitive example data, predictable matches for known topics, and an explicit result for unknown topics.

#### Scenario: Known documentation topic
- **WHEN** `search_docs` receives a query containing a documented known topic
- **THEN** it returns the corresponding fixed policy text

#### Scenario: Unknown documentation topic
- **WHEN** `search_docs` receives a query that matches no documented topic
- **THEN** it returns the documented not-found result without raising an unhandled exception

### Requirement: Equivalent LangGraph agent loop
The graph lesson SHALL reproduce the observable behavior of the manual tool loop using explicit message state, a model node, a tool-execution node, conditional routing, a return edge from tools to the model, and an end condition.

#### Scenario: Graph routes to a tool
- **WHEN** the model node returns an assistant message containing one or more tool calls
- **THEN** the graph routes to tool execution and returns the resulting tool messages to the model node

#### Scenario: Graph terminates on final answer
- **WHEN** the model node returns an assistant message without tool calls
- **THEN** the graph terminates and exposes that message as the final response

### Requirement: Conversational state and local persistence
The learning path SHALL demonstrate accumulated message state across multiple turns and SHALL separately demonstrate checkpointed thread state using an explicit thread identifier. The documentation SHALL state that the learning-path checkpointer is process-local and not durable across restarts.

#### Scenario: Continue an existing thread
- **WHEN** two graph invocations use the same configured thread identifier within one process
- **THEN** the second invocation has access to the checkpointed messages from the first invocation

#### Scenario: Isolate different threads
- **WHEN** graph invocations use different thread identifiers
- **THEN** messages from one thread are not included in the state of the other thread

### Requirement: Optional LangSmith tracing
The learning path SHALL include an opt-in tracing lesson that reuses the working graph and documents the required LangSmith environment configuration. Core lessons and default tests SHALL function without LangSmith credentials or tracing enabled.

#### Scenario: Tracing is disabled
- **WHEN** LangSmith configuration is absent
- **THEN** all non-tracing lessons and the default offline test suite remain usable

#### Scenario: Tracing is enabled
- **WHEN** valid LangSmith tracing configuration is present and the tracing lesson runs
- **THEN** the run produces an inspectable trace containing model and tool execution steps without changing the agent's functional result

### Requirement: Bounded and understandable failures
Agent execution SHALL have a finite iteration limit and SHALL present understandable errors for missing live credentials, unknown requested tools, invalid tool arguments, and tool execution failures. Educational examples SHALL not expose credential values in output.

#### Scenario: Repeated tool requests reach the limit
- **WHEN** a scripted model continues requesting tools beyond the configured execution limit
- **THEN** execution stops with a clear bounded-execution error instead of looping indefinitely

#### Scenario: Live credentials are missing
- **WHEN** a learner starts a live-model lesson without the required provider credentials
- **THEN** the lesson exits with setup guidance and does not print any secret values

### Requirement: Offline verification and evaluation
The project SHALL provide a default test suite that validates deterministic tools, message construction, routing, state isolation, and execution limits without network access. It SHALL also provide a small evaluation dataset that exercises known, unknown, tool-needed, and tool-not-needed cases and reports results in a comparable form.

#### Scenario: Run default tests without credentials
- **WHEN** the default automated test command runs in an environment without provider or LangSmith credentials
- **THEN** deterministic tests complete without attempting external model or tracing requests

#### Scenario: Evaluate representative cases
- **WHEN** the evaluation lesson runs against its documented dataset
- **THEN** it records an outcome for every case and distinguishes at least answer correctness and expected tool-use behavior

### Requirement: Learner-facing project documentation
The root README SHALL be a concise project entry point that documents the project goal, prerequisites, supported Python setup, dependency installation, environment variables, lesson order, commands, expected observations, test commands, optional tracing, cost and privacy considerations, common troubleshooting, and links to deeper documentation. The primary README SHALL not contain raw chat-export controls or rely on the source conversation as operational documentation.

The project SHALL provide `docs/system-overview.md`, which SHALL transform the useful conceptual content from the source conversation into standalone reference documentation covering the roles and relationships of LLMs, tools, LangChain, LangGraph, and LangSmith. It SHALL not retain transcript structure, timestamps, speaker labels, chat UI controls, or conversational filler.

#### Scenario: New learner prepares the project
- **WHEN** a learner follows the README on a clean checkout
- **THEN** the learner can install dependencies, configure a live provider when needed, run the first lesson, and run offline tests using the documented commands

#### Scenario: Learner reads the system overview
- **WHEN** a learner follows the conceptual-overview link from the README
- **THEN** `docs/system-overview.md` explains the system components and agent flow as a coherent article without requiring knowledge of the original conversation
