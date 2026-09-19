"""Lesson 03: inspect a model's request to use a tool without executing it."""

from langchain_core.messages import HumanMessage

from agentsmith.agent import bind_tools
from agentsmith.tools import search_docs
from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        model = bind_tools(configured_model(live=args.live), [search_docs])
        response = model.invoke(
            [HumanMessage(content="How many vacation days do employees receive?")]
        )
        print(f"Assistant content: {response.content!r}")
        print("The model requested; Python has not executed anything yet:")
        for call in response.tool_calls:
            print(f"  name={call['name']} args={call['args']} id={call['id']}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
