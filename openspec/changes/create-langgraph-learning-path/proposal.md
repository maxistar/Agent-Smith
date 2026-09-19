## Why

The repository currently contains only a Python scaffold and a raw introductory conversation, so it does not yet provide an executable path for learning how an LLM call becomes a stateful LangGraph agent. A progressive sequence of small, independently runnable examples will make each abstraction observable before the next one is introduced.

## What Changes

- Add a numbered learning path that progresses from a basic model invocation to tool declaration, model-requested tool calls, manual tool execution, a LangGraph agent loop, conversational state, persistence, LangSmith tracing, reliability controls, and evaluation.
- Keep every lesson independently runnable and focused on one primary new concept, with expected behavior documented so learners can compare stages.
- Use a small deterministic `search_docs` knowledge tool as the shared domain across lessons, keeping attention on orchestration rather than application complexity.
- Replace the raw README transcript with a focused project entry point, and transform the useful content from the conversation into a coherent system overview under `docs/`.
- Document environment setup, model credentials, optional LangSmith configuration, lesson commands, and conceptual diagrams in the focused project README.
- Add lightweight automated checks for deterministic components and graph routing without requiring live paid API calls by default.
- Exclude voice interaction and production deployment from this learning path; the voice command in the source conversation controlled the lecture and is not an application requirement.

## Capabilities

### New Capabilities

- `langgraph-learning-path`: A progressive, independently runnable curriculum demonstrating model messages, tools, LangGraph orchestration, state, persistence, observability, reliability, and evaluation.

### Modified Capabilities

None.

## Impact

- Adds Python dependencies for LangChain, LangGraph, an LLM provider integration, environment configuration, and test tooling; LangSmith remains optional at runtime.
- Replaces the placeholder entry point and raw README transcript with a focused lesson-oriented README, a rewritten system overview under `docs/`, and executable examples.
- Introduces example code, shared fixtures or utilities where useful, and tests that separate deterministic validation from optional live integration checks.
- Requires users to provide model-provider credentials for live lessons and LangSmith credentials only for tracing lessons.
