"""Lesson 05: express the manual loop as an explicit LangGraph state machine."""

from langchain_core.messages import HumanMessage

from agentsmith.agent import build_agent_graph, format_trajectory, invoke_agent
from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        graph = build_agent_graph(configured_model(live=args.live))
        result = invoke_agent(
            graph,
            [HumanMessage(content="How many vacation days do employees receive?")],
        )
        print(format_trajectory(result["messages"]))
        print("\nGraph route: START -> agent -> tools -> agent -> END")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
