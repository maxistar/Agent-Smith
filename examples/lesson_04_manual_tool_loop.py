"""Lesson 04: execute the model's tool request explicitly in application code."""

from langchain_core.messages import HumanMessage

from agentsmith.agent import format_trajectory, run_manual_tool_loop
from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        messages = [HumanMessage(content="How many vacation days do employees receive?")]
        history = run_manual_tool_loop(configured_model(live=args.live), messages)
        print(format_trajectory(history))
        print("\nNotice: the application inserted ToolMessage between two model calls.")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
