"""Replay: stream recorded results into the board as if a room of runners produced them.

Offline fallback (no network needed): virtual runners claim the cells in the grid's claim order,
each "runs" for its recorded wall time, and the whole schedule is squeezed into `minutes`. The
events go through the same Store as live claims, so the board cannot tell the difference except
for the REPLAY badge and the runner names (`replay-000`, ...).
"""

from __future__ import annotations

import heapq
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import Conflict, Store

MIN_WALL_S = 30.0


def load_results(path: str | Path) -> list[dict[str, Any]]:
    doc = json.loads(Path(path).read_text())
    return doc["results"] if isinstance(doc, dict) else doc


@dataclass(frozen=True)
class Event:
    at: float  # seconds after the replay start (real time)
    kind: str  # "claim" | "submit"
    cell_id: str
    user: str


class Ticker:
    """Runs `step()` every tick in a daemon thread until `done` or `stop()`. Subclasses set `_stop` and `name`."""

    name = "replay"
    _stop: threading.Event

    @property
    def done(self) -> bool:
        raise NotImplementedError

    def step(self) -> int:
        raise NotImplementedError

    def run(self, tick_s: float = 0.5) -> None:
        while not self.done and not self._stop.wait(tick_s):
            self.step()

    def start_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name=self.name, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()


class Replayer(Ticker):
    def __init__(self, store: Store, results: list[dict[str, Any]], *, minutes: float = 20.0, concurrency: int = 150,
                 clock: Callable[[], float] = time.time):
        self.store, self.clock = store, clock
        self.results = {r["cell_id"]: r for r in results}
        order = [c.cell_id for c in store.grid.cells if c.cell_id in self.results]
        runners = [(0.0, i) for i in range(min(concurrency, max(len(order), 1)))]
        virtual: list[tuple[float, str, str, str]] = []
        for cid in order:
            free_at, i = heapq.heappop(runners)
            end = free_at + max(float(self.results[cid].get("wall_s") or 0.0), MIN_WALL_S)
            user = f"replay-{i:03d}"
            virtual += [(free_at, "claim", cid, user), (end, "submit", cid, user)]
            heapq.heappush(runners, (end, i))
        makespan = max((v[0] for v in virtual), default=0.0) or 1.0
        scale = minutes * 60.0 / makespan
        # at equal times a submit sorts before a claim, the order a real runner follows (finish, then claim again)
        self.events = sorted((Event(round(t * scale, 3), k, c, u) for t, k, c, u in virtual), key=lambda e: (e.at, e.kind != "submit"))
        self.start = clock()
        self._next = 0
        self._stop = threading.Event()

    @property
    def done(self) -> bool:
        return self._next >= len(self.events)

    def step(self) -> int:
        """Apply every event that is due. Returns how many were applied."""
        now = self.clock() - self.start
        n = 0
        while self._next < len(self.events) and self.events[self._next].at <= now:
            e = self.events[self._next]
            self._next += 1
            n += 1
            try:
                if e.kind == "claim":
                    self.store.lease(e.cell_id, e.user)
                else:
                    self.store.submit(e.cell_id, e.user, {**self.results[e.cell_id], "hf_user": e.user})
            except Conflict:
                pass  # the cell already holds a result (a replay into a live store); skip it
        return n
