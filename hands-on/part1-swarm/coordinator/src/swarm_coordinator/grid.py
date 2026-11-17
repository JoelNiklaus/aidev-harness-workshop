"""The trial grid (config/grid.json) and the order in which its cells are handed out.

Order (Joel, 17 Nov 2026; PROTOCOL.md): one harness after the next, in grid.json `harnesses`
order; within a harness breadth-first over trials (every task gets trial 0 before any gets
trial 1), tasks in grid.json order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# grid.json keys that describe the grid itself rather than how to run a cell
_GRID_KEYS = {"model", "harnesses", "tasks", "trials_per_cell"}


def cell_id(harness: str, task: str, trial: int) -> str:
    """`<harness>__<task>__<trial>`: safe in URLs and file names (harness and task names hold no `__`)."""
    return f"{harness}__{task}__{trial}"


@dataclass(frozen=True)
class Cell:
    harness: str
    task: str
    trial: int

    @property
    def cell_id(self) -> str:
        return cell_id(self.harness, self.task, self.trial)


@dataclass(frozen=True)
class Grid:
    model: str
    harnesses: tuple[str, ...]
    tasks: tuple[str, ...]
    trials_per_cell: int
    run_config: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Grid:
        if any("__" in x for x in [*d["harnesses"], *d["tasks"]]):
            raise ValueError("harness and task names must not contain '__'")
        rc = {k: v for k, v in d.items() if k not in _GRID_KEYS and not k.startswith("_")}
        rc["trials_per_cell"] = d["trials_per_cell"]
        return cls(d["model"], tuple(d["harnesses"]), tuple(d["tasks"]), int(d["trials_per_cell"]), rc)

    @classmethod
    def load(cls, path: str | Path) -> Grid:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def cells(self) -> list[Cell]:
        return [Cell(h, t, k) for h in self.harnesses for k in range(self.trials_per_cell) for t in self.tasks]
