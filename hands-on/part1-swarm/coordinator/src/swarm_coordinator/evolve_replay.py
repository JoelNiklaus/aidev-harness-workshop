"""Evolve replay: stream recorded evolve runs into the /evolve board as if attendees' loops sent them.

Offline fallback (no network): each recorded event is posted at its recorded time (seconds after its run
started), squeezed so the longest run takes `minutes`. Events go through the same EvolveStore and validation
as live submissions; the board shows a REPLAY badge and the users `recorded-B1`, `recorded-B2`.
In live mode the same recording is the race chart's backdrop (`reference_curves`).

Recording format (`replay/evolve-recorded.json`, built by `replay/build_evolve_replay.py`):
{"source": str, "runs": [{"name", "user", "events": [{"t": seconds, ...one aidev-evolve/1 event}]}]}
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import evolve
from .replay import Ticker


def load_recorded(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())["runs"]


def reference_curves(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each recorded run's race curve: the seed's validation score at 0, then best_val per iteration."""
    out = []
    for r in runs:
        curve = [[0, e["val_score"]] if e["kind"] == "seed" else [e["iteration"], e["best_val"]]
                 for e in r["events"] if e["kind"] in ("seed", "iteration")]
        out.append({"label": f"recorded {r['name']}", "curve": curve})
    return out


class EvolveReplayer(Ticker):
    name = "evolve-replay"

    def __init__(self, store: evolve.EvolveStore, runs: list[dict[str, Any]], *, minutes: float = 20.0,
                 clock: Callable[[], float] = time.time):
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        self.store, self.clock = store, clock
        span = max((e["t"] for r in runs for e in r["events"]), default=0.0) or 1.0
        scale = minutes * 60.0 / span
        events = [(e["t"] * scale, r["user"], {k: v for k, v in e.items() if k != "t"}) for r in runs for e in r["events"]]
        self.events = sorted(events, key=lambda x: x[0])
        self.start = clock()
        self._next = 0
        self._stop = threading.Event()

    @property
    def done(self) -> bool:
        return self._next >= len(self.events)

    def step(self) -> int:
        """Post every event that is due. Returns how many were posted."""
        now = self.clock() - self.start
        n = 0
        while self._next < len(self.events) and self.events[self._next][0] <= now:
            _, user, event = self.events[self._next]
            self._next += 1
            self.store.add(user, evolve.validate(event))
            n += 1
        return n
