"""Lesson 01: inspect the message returned by a chat model."""

from langchain_core.messages import HumanMessage

from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        model = configured_model(live=args.live)
        prompt = HumanMessage(content="Hello! Answer in one short sentence.")
        response = model.invoke([prompt])
        print(f"Input type: {type(prompt).__name__}")
        print(f"Output type: {type(response).__name__}")
        print(f"Output content: {response.content}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
