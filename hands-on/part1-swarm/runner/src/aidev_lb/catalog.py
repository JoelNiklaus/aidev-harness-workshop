"""The packaged data files: the harnesses with their speed labels, the ordered tasks, prices."""

from __future__ import annotations

import json
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any


def data(name: str) -> Traversable:
    return resources.files("aidev_lb").joinpath("data", name)


def _json(name: str) -> Any:
    return json.loads(data(name).read_text())


def harnesses() -> list[dict[str, str]]:
    """[{"name", "speed"}]: the harnesses `run --harness` accepts, fastest install first."""
    return _json("harnesses.json")["harnesses"]


def spec() -> dict[str, Any]:
    """Model, benchmark, reasoning effort, the ordered task list and the default task count."""
    return _json("tasks.json")


def tasks(n: int | None = None) -> list[str]:
    """The first `n` tasks of the ordered list (default: the five of the workshop)."""
    s = spec()
    n = s["default_n"] if n is None else n
    if not 1 <= n <= len(s["tasks"]):
        raise ValueError(f"the task count must be from 1 to {len(s['tasks'])}")
    return s["tasks"][:n]


def prices() -> dict[str, dict[str, Any]]:
    return _json("prices.json")


def cell_id(harness: str, task: str, trial: int = 0) -> str:
    return f"{harness}__{task}__{trial}"
