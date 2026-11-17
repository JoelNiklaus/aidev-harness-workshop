from __future__ import annotations

import pytest

from swarm_coordinator.grid import Grid
from swarm_coordinator.store import Store

GRID = {
    "model": "zai-org/GLM-5.3-Flash",
    "harnesses": ["opencode", "openclaw", "pi"],
    "tasks": ["t1", "t2"],
    "trials_per_cell": 2,
    "benchmark": "terminal-bench-2.1",
    "reasoning_effort": "low",
    "provider": "zai-org",
    "_comment": "test grid",
}
PRICES = {"zai-org/GLM-5.3-Flash": {"input": 0.15, "cached_input": 0.03, "output": 0.5, "unit": "USD per 1M tokens"}}


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class MemorySinkSpy:
    """Records what the store hands to its sink."""

    def __init__(self, results=None, leases=None):
        self.results: dict[str, dict] = {r["cell_id"]: r for r in (results or [])}
        self.leases: list[dict] = list(leases or [])
        self.put_calls = 0

    def put(self, result: dict) -> None:
        self.put_calls += 1
        self.results[result["cell_id"]] = result

    def put_leases(self, leases: list[dict]) -> None:
        self.leases = list(leases)

    def load(self):
        return list(self.results.values()), list(self.leases)


def make_result(cell_id: str, user: str = "alice", passed: bool = True, **kw) -> dict:
    h, t, k = cell_id.split("__")
    r = {
        "cell_id": cell_id, "harness": h, "harness_version": "1.0", "task": t, "trial": int(k),
        "model": GRID["model"], "reward": 1.0 if passed else 0.0, "passed": passed,
        "started_at": "2026-11-30T15:00:00+00:00", "finished_at": "2026-11-30T15:10:00+00:00", "wall_s": 600.0,
        "n_input_tokens": 1_000_000, "n_cached_tokens": 500_000, "n_output_tokens": 10_000, "n_calls": 20,
        "cost_usd": 99.0, "error": None, "hf_user": user, "runner_version": "0.1.0",
        "sandbox_id": None, "trajectory_url": None,
    }
    r.update(kw)
    return r


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def grid():
    return Grid.from_dict(GRID)


@pytest.fixture
def sink():
    return MemorySinkSpy()


@pytest.fixture
def store(grid, clock, sink):
    return Store(grid, PRICES, clock=clock, sink=sink)
