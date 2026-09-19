## 1. Project Foundation

- [x] 1.1 Verify LangChain, LangGraph, provider integration, and test dependencies against the current Python 3.14 baseline; update the declared Python version only if compatibility requires it.
- [x] 1.2 Add the selected runtime and development dependencies with a reproducible project lockfile.
- [x] 1.3 Create the importable `agentsmith` package, numbered `examples` layout, and `tests` layout needed by the learning path.
- [x] 1.4 Add environment-based model configuration with a documented default provider/model and a clear, secret-safe error when credentials are missing.
- [x] 1.5 Add an ignored environment template containing provider, model, and optional LangSmith variable names without real credentials.

## 2. Shared Deterministic Tool

- [x] 2.1 Implement the fixed example policy data and the documented `search_docs` tool contract for known and unknown topics.
- [x] 2.2 Add offline tests for known-topic matching, case normalization, unknown-topic behavior, and invalid inputs.
- [x] 2.3 Create lesson 02 to invoke `search_docs` directly and display its input and output without using an LLM.

## 3. Model and Manual Tool Lessons

- [x] 3.1 Create lesson 01 to invoke the configured chat model and display the returned assistant message type and content.
- [x] 3.2 Create lesson 03 to bind `search_docs`, request a tool-relevant answer, and display the requested tool name, arguments, and call identifier without executing it.
- [x] 3.3 Create lesson 04 to execute requested tools in application code, append correctly associated tool-result messages, and obtain the final model response.
- [x] 3.4 Add scripted-model offline tests for tool-request inspection, tool dispatch, matching call identifiers, unknown tools, invalid arguments, and tool failures.

## 4. LangGraph Orchestration Lessons

- [x] 4.1 Implement reusable low-level graph construction with message state, an explicit model node, tool execution, conditional routing, a return edge, and termination.
- [x] 4.2 Create lesson 05 to run the graph loop and expose its intermediate message trajectory alongside the final answer.
- [x] 4.3 Add offline graph tests proving tool-call routing, return-to-model behavior, and termination when the model returns no tool calls.
- [x] 4.4 Create lesson 06 to perform multiple conversational turns and demonstrate accumulated message state.
- [x] 4.5 Create lesson 07 with an in-memory checkpointer and explicit thread identifiers, demonstrating same-thread continuity and cross-thread isolation.
- [x] 4.6 Add offline state tests for accumulated messages, same-thread resume, different-thread isolation, and the documented loss of in-memory state after reconstruction.

## 5. Observability, Reliability, and Evaluation

- [x] 5.1 Create lesson 08 to enable optional LangSmith tracing for the existing graph without changing its functional result.
- [x] 5.2 Document and verify that core examples and default tests do not require LangSmith configuration or make tracing requests when tracing is disabled.
- [x] 5.3 Add a finite graph execution limit and learner-readable handling for repeated tool requests, unknown tools, invalid arguments, and tool exceptions.
- [x] 5.4 Create lesson 09 to demonstrate each reliability boundary with safe example inputs and observable outcomes.
- [x] 5.5 Add offline tests that force repeated tool requests and verify bounded termination and secret-safe error output.
- [x] 5.6 Define a small evaluation dataset covering known topics, unknown topics, tool-required questions, and questions answerable without a tool.
- [x] 5.7 Create lesson 10 to run the dataset and report comparable answer-correctness and expected-tool-use outcomes for every case.

## 6. Learner Documentation

- [x] 6.1 Move the useful content from the README conversation into `docs/system-overview.md` and rewrite it as coherent general documentation about LLMs, tools, LangChain, LangGraph, LangSmith, agent flow, tracing, and evaluation, removing transcript and chat-UI artifacts.
- [x] 6.2 Replace the raw transcript with a correctly named, focused `README.md` that introduces the project, documents prerequisites, installation, environment setup, credential safety, lesson order, and exact commands, and links to the system overview.
- [x] 6.3 Document expected observations for every lesson, emphasizing the distinction between a model tool request, application-side execution, and LangGraph orchestration.
- [x] 6.4 Add diagrams for the manual tool loop, graph routing, conversation state, and tracing trajectory.
- [x] 6.5 Document offline tests, optional live checks, LangSmith setup, cost/privacy considerations, in-memory persistence limits, and common troubleshooting.

## 7. End-to-End Verification

- [x] 7.1 Run formatting, static checks, and the complete offline test suite with provider and LangSmith credentials absent.
- [x] 7.2 Execute every numbered lesson using its documented command, using scripted paths where available and explicitly recording which lessons require opt-in live credentials.
- [ ] 7.3 Perform opt-in smoke checks for the configured live model, tool calling, and LangSmith tracing without committing secrets or generated trace data.
- [x] 7.4 Review the completed learning path against every scenario in `specs/langgraph-learning-path/spec.md` and resolve any documentation or behavior gaps.
