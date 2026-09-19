"""Lesson 07: checkpoint state by thread id within one process."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentsmith.agent import build_agent_graph, invoke_agent
from examples._shared import configured_model, lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        graph = build_agent_graph(
            configured_model(live=args.live),
            checkpointer=InMemorySaver(),
        )
        first = invoke_agent(
            graph,
            [HumanMessage(content="What is the vacation policy?")],
            thread_id="learner-a",
        )
        continued = invoke_agent(
            graph,
            [HumanMessage(content="What is the remote-work policy?")],
            thread_id="learner-a",
        )
        isolated = invoke_agent(
            graph,
            [HumanMessage(content="Hello")],
            thread_id="learner-b",
        )
        print(f"Thread learner-a after turn 1: {len(first['messages'])} messages")
        print(f"Thread learner-a after turn 2: {len(continued['messages'])} messages")
        print(f"Thread learner-b: {len(isolated['messages'])} messages")
        print("InMemorySaver state disappears when this Python process exits.")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
