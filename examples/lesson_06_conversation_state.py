"""Lesson 06: carry accumulated message state into the next turn."""

from langchain_core.messages import HumanMessage

from agentsmith.agent import build_agent_graph, format_trajectory, invoke_agent
from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        graph = build_agent_graph(configured_model(live=args.live))
        first = invoke_agent(graph, [HumanMessage(content="What is the vacation policy?")])
        second_input = [*first["messages"], HumanMessage(content="And the remote-work policy?")]
        second = invoke_agent(graph, second_input)
        print(format_trajectory(second["messages"]))
        print(f"\nMessages after two turns: {len(second['messages'])}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
