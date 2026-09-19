"""Lesson 08: observe the existing graph with optional LangSmith tracing."""

import os

from langchain_core.messages import HumanMessage

from agentsmith.agent import build_agent_graph, invoke_agent
from agentsmith.models import require_langsmith_configuration
from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        if args.live:
            require_langsmith_configuration()
        else:
            print("Offline mode: no trace is uploaded. Add --live after configuring LangSmith.")

        graph = build_agent_graph(configured_model(live=args.live))
        result = invoke_agent(
            graph,
            [HumanMessage(content="What is the vacation policy?")],
        )
        print(f"Final answer: {result['messages'][-1].content}")
        if args.live:
            project = os.getenv("LANGSMITH_PROJECT", "default")
            print(f"Trace sent to LangSmith project: {project}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
