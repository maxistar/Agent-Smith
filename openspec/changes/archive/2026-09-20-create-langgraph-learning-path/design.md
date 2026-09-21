## Context

The repository is a fresh Python project with a placeholder entry point, no dependencies, no tests, and a README containing the source conversation that motivated learning LangChain, LangGraph, and LangSmith. The intended audience is a backend-oriented learner who benefits from seeing control flow, state, and observability explicitly rather than starting with a high-level prebuilt agent.

The learning path must distinguish model output from application-side tool execution, then show how LangGraph formalizes the same manually constructed loop. Live model calls require external credentials and may incur cost, while most structural behavior can be taught and tested deterministically without network access. The current project targets Python 3.14; dependency compatibility must be verified during implementation before changing that baseline.

## Goals / Non-Goals

**Goals:**

- Teach one primary concept per lesson in a deliberate progression from model invocation through evaluation.
- Keep each lesson directly runnable and make its inputs, intermediate messages, and outputs visible.
- Preserve a stable example domain so differences between lessons come from orchestration rather than changing business logic.
- Demonstrate the manual tool loop before expressing it as a LangGraph state machine.
- Provide deterministic automated coverage for tools, routing, state updates, and failure limits without requiring live credentials.
- Make LangSmith tracing opt-in and show how it explains the trajectory of an otherwise unchanged run.

**Non-Goals:**

- Building a production knowledge base, RAG pipeline, web service, GUI, or voice interface.
- Teaching multi-agent coordination, deployment, streaming, authentication, or production-grade persistence.
- Hiding orchestration behind a prebuilt agent abstraction before the learner understands the underlying messages and transitions.
- Requiring LangSmith or any external tracing service for the core lessons to execute.

## Decisions

### Use numbered executable lessons

Place lessons in a numerically ordered `examples/` collection. Each lesson is executable from the repository root and introduces one primary concept while retaining concepts learned earlier. Expected lesson progression is:

1. invoke a chat model and inspect the returned message;
2. invoke the deterministic documentation tool directly;
3. bind the tool and inspect a model-generated `tool_calls` request;
4. execute requested tools manually and append matching tool-result messages;
5. express the same loop using LangGraph nodes and conditional edges;
6. continue a multi-turn conversation using message state;
7. preserve thread state with a local in-memory checkpointer;
8. enable and inspect LangSmith tracing without changing graph behavior;
9. handle unknown queries, tool errors, and iteration limits;
10. run a small evaluation dataset and report comparable outcomes.

This is preferred over repeatedly replacing `main.py`, because learners can run and compare earlier stages after the final stage exists. It is also preferred over relying only on Git history, which makes lessons less discoverable and harder to execute side by side.

### Keep the domain deterministic and shared

Use one small `search_docs` tool backed by fixed in-repository data such as vacation and remote-work policies. Later lessons reuse the same tool contract. Shared deterministic components may live in a small importable project package, while orchestration remains visible in each lesson.

This avoids introducing retrieval infrastructure before tool-calling behavior is understood. A real vector store or external search API would add credentials, nondeterminism, and unrelated failure modes.

### Use explicit low-level graph primitives

Construct the agent lesson with message state, a model node, a tool node, conditional routing, and an explicit termination path. Do not begin with a high-level agent factory. The manual-loop lesson and graph lesson must have equivalent observable behavior so the learner can identify exactly what LangGraph contributes.

The graph must terminate when the model returns no tool calls and must have a finite execution limit for abnormal repeated tool requests.

### Centralize model configuration without hiding lesson behavior

Use a small model factory configured through environment variables for provider/model selection and credentials. The default documented provider follows the OpenAI-based source example, but lesson logic must depend on standard LangChain chat-model and tool-call interfaces rather than provider-specific response fields.

Credentials must never be committed. Missing credentials should result in a concise setup error for live lessons. Model selection must not be duplicated across every example.

### Separate offline tests from opt-in live checks

Deterministic tests use fixed tool inputs and scripted or fake model responses to verify message construction, routing, loop termination, state continuity, and error handling. Live provider and LangSmith checks are opt-in and clearly marked because they require secrets, network access, and may cost money.

This preserves a fast default test suite while still allowing learners to validate the real integrations deliberately.

### Introduce persistence locally before external storage

Teach persistence with a local in-memory checkpointer and explicit thread identifiers. This demonstrates the semantics of resumable state without adding a database. Durable cross-process storage is deferred beyond this change.

### Treat LangSmith as an observational layer

The tracing lesson reuses an already working graph and enables tracing through documented environment configuration. It explains model calls, tool calls, latency, errors, and token usage where available. Core behavior must remain the same with tracing disabled.

### Separate the project entry point from the conceptual overview

Rewrite the root README as a concise, goal-oriented entry point for the project: what the learner will build, prerequisites, quick start, lesson map, run and test commands, and links to deeper documentation. It must help a new learner begin without reading the history of how the project idea emerged.

Move the useful substance of the source conversation into `docs/system-overview.md`, but rewrite it as standalone reference material rather than preserving the question-and-answer transcript. The overview will explain the roles and relationships of LLMs, tools, LangChain, LangGraph, and LangSmith; illustrate the manual and graph-based agent loops; and introduce tracing and evaluation. Chat timestamps, speaker labels, generated UI controls, conversational filler, and duplicated instructions will be removed.

## Risks / Trade-offs

- **[Repeated concepts across numbered examples]** → Accept modest duplication so every stage remains readable; factor out only stable infrastructure such as the tool data and model factory.
- **[LLM output is nondeterministic]** → Assert structural properties in tests and use scripted model responses; present live outputs as examples, not exact golden text.
- **[Provider behavior differs for tool choice]** → Document a tested default provider and depend only on standardized LangChain message/tool-call shapes in lesson code.
- **[Python 3.14 dependency incompatibility]** → Resolve and exercise the selected packages early; adjust the declared Python baseline only if the current ecosystem cannot support the scaffolded version.
- **[External calls create cost or expose prompts]** → Make live and tracing lessons opt-in, document data flow, avoid sending secrets or sensitive example data, and keep offline tests as the default.
- **[Too many lessons dilute the central agent loop]** → Keep lessons short, state one learning objective per lesson, and make lessons 3–5 the conceptual core of the README.
- **[In-memory persistence may be mistaken for durable storage]** → Explicitly demonstrate that state survives calls within a process but not a process restart.
