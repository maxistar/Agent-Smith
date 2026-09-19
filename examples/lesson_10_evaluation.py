"""Lesson 10: compare answer correctness and expected tool use across a dataset."""

from pathlib import Path

from agentsmith.evaluation import evaluate_cases, load_cases
from examples._shared import lesson_parser, run_lesson


def main() -> None:
    args = lesson_parser(__doc__).parse_args()

    def lesson() -> None:
        cases_path = Path(__file__).parents[1] / "data" / "evaluation_cases.json"
        results = evaluate_cases(load_cases(cases_path), live=args.live)
        print("case                 answer  tool-use  overall")
        print("-------------------  ------  --------  -------")
        for result in results:
            print(
                f"{result.case_id:19}  "
                f"{str(result.answer_correct):6}  "
                f"{str(result.tool_use_correct):8}  "
                f"{str(result.passed):7}"
            )
        passed = sum(result.passed for result in results)
        print(f"\nPassed: {passed}/{len(results)}")

    run_lesson(lesson)


if __name__ == "__main__":
    main()
