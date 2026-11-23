"""The seed agent's scores, measured once before the workshop and shipped in the wheel.

The seed is the same file for every attendee, and so are the tasks, the model and the time budget. Measuring it again
in every run cost each attendee a Job of 3 to 8 minutes under room load (PROGRESS.md, 23 Nov) and told nobody anything
new. A calibration (`data/seed_calibration.json`, written by `devtools/seed_calibration.py`) holds:

- `search_sets`: several independent trials of the seed on the search tasks, with their trajectories. A run reads one
  set, picked by its run id, so the room's proposers do not all read the same five trajectories;
- `validation`: the seed's validation trials (outcomes only, no trajectories: the proposer never sees validation). They
  give the flip rate behind the keep margin, and the parent's pass rate pools them with the trials of the run's own Job.

It is valid only for the seed and the time budget it was measured with (`check`); a run with its own seed, splits or
budget measures the seed itself, as `run --measure-seed` and `run --full` do.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from importlib import resources
from typing import Any

SEED = "c000"
FILE = "seed_calibration.json"


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def build(records: list[dict[str, Any]], splits: dict[str, Any], *, seed_source: str, budget_s: float, sets: int, measured: dict[str, Any]) -> dict[str, Any]:
    """A calibration from the seed's scored trials: `sets` search sets (the i-th scored trial of every search task) and
    every validation trial without its trajectory. Refuses a measurement that does not cover the splits."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["outcome"] != "infra":
            by_task[r["task"]].append({**r, "candidate": SEED})
    short = [t for t in splits["search"] if len(by_task[t]) < sets] + [t for t in splits["validation"] if len(by_task[t]) < 2]
    if short:
        raise ValueError(f"too few scored seed trials for {', '.join(short)}: {sets} per search task and 2 per validation task are needed")
    if any(not r.get("trajectory") for t in splits["search"] for r in by_task[t][:sets]):
        raise ValueError("a search trial has no trajectory: the proposer would have nothing to read")
    return {"seed_sha256": _sha(seed_source), "budget_s": budget_s, "measured": measured,
            "search_sets": [[by_task[t][i] for t in splits["search"]] for i in range(sets)],
            "validation": [{k: v for k, v in r.items() if k != "trajectory"} for t in splits["validation"] for r in by_task[t]]}


def bundled() -> dict[str, Any]:
    return json.loads(resources.files("aidev_evolve.data").joinpath(FILE).read_text())


def check(cal: dict[str, Any], seed_source: str, budget_s: float) -> None:
    if cal["seed_sha256"] != _sha(seed_source):
        raise ValueError("the shipped seed scores were measured with another seed agent: measure this one (run --measure-seed)")
    if float(cal["budget_s"]) != float(budget_s):
        raise ValueError(f"the shipped seed scores were measured with a time budget of {cal['budget_s']:g} s, not {budget_s:g} s: run --measure-seed")


def pick(cal: dict[str, Any], run_id: str) -> int:
    """Which search set a run reads: fixed by its run id."""
    return int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big") % len(cal["search_sets"])


def records(cal: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    """The seed's trials as a run uses them: one search set and every validation trial."""
    return [*cal["search_sets"][pick(cal, run_id)], *cal["validation"]]
