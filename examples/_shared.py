"""Small CLI helpers shared by the numbered lessons."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from agentsmith.models import ModelConfigurationError, select_model


def lesson_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use OpenAI. Without this flag, use a deterministic offline model.",
    )
    return parser


def configured_model(*, live: bool):
    mode = "LIVE model (network/cost may apply)" if live else "OFFLINE deterministic model"
    print(f"Mode: {mode}")
    return select_model(live=live)


def run_lesson(action: Callable[[], None]) -> None:
    try:
        action()
    except ModelConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
