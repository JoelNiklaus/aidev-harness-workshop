"""Offline fallback: replay a recorded evolve run's terminal output with its real rhythm, no network.

The loop writes every terminal line to `transcript.jsonl` with its time; replay prints them again, pauses
scaled by `speed` and capped, so a 30-minute run plays in a few minutes. A real recorded run ships in the
package (`data/recorded/`), so `aidev-evolve replay` works on a laptop with no Wi-Fi and no credits.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path


def bundled() -> Path:
    return Path(str(resources.files("aidev_evolve.data").joinpath("recorded")))


def replay(run_dir: Path, *, speed: float = 20.0, max_pause_s: float = 4.0, write: Callable[[str], None] = lambda s: print(s, end="", flush=True),
           sleep: Callable[[float], None] = time.sleep) -> None:
    last = None
    for line in (Path(run_dir) / "transcript.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if last is not None:
            pause = min(max_pause_s, (ev["t"] - last) / speed)
            if pause > 0:
                sleep(pause)
        last = ev["t"]
        write(ev["text"])
