from pathlib import Path

from agentsmith.evaluation import evaluate_cases, load_cases


def test_offline_evaluation_covers_and_passes_representative_cases() -> None:
    cases = load_cases(Path("data/evaluation_cases.json"))
    assert {case["id"] for case in cases} == {
        "known-vacation",
        "known-remote",
        "unknown-policy",
        "no-tool-needed",
    }

    results = evaluate_cases(cases)
    assert all(result.passed for result in results)
