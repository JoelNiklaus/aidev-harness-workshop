"""Fill-time model of the 600-cell grid (runner/PROGRESS.md 2026-11-21 table): attendees x slots, the reference
layer's trial walls (board GLM-5.3-Flash low, TB2.1 subset, incl. sandbox start and harness install) + 17 s per trial,
the coordinator's share and slow start, the runner's passive slots and probe backoff. Arrivals over 3 min, Job start 1 min.

    python3 devtools/fill_model.py
"""

from __future__ import annotations

import heapq
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = {r["cell_id"]: r["wall_s"] for r in json.loads((ROOT / "replay" / "reference.json").read_text())["results"]}
GRID = json.loads((ROOT / "runner" / "src" / "aidev_swarm" / "data" / "grid.json").read_text())
CELLS = [f"{h}__{t}__{k}" for h in GRID["harnesses"] for t in GRID["tasks"] for k in range(GRID.get("trials", 3))]
OVER_S = 17.0  # claim + Harbor job + submit around a trial (staging run: 116 s claim->submit for a 99 s trial)


def sim(n_att: int, slots: int, seed: int = 0, arrive_s: float = 180, startup_s: float = 60, ramp_s: float = 20) -> dict[str, float]:
    rng = random.Random(seed)
    ev: list[tuple[float, int, int]] = []
    first: dict[int, float] = {}
    seen: dict[int, float] = {}
    known, refus = [2] * n_att, {}
    for a in range(n_att):
        t0 = rng.uniform(0, arrive_s) + startup_s
        for s in range(slots):
            heapq.heappush(ev, (t0, a, s))
    held, i, ends, fin, claimed = [0] * n_att, 0, [], [], 0.0
    while i < len(CELLS):
        t, a, s = heapq.heappop(ev)
        while fin and fin[0][0] <= t:
            held[heapq.heappop(fin)[1]] -= 1
        if slots > 1 and s > known[a]:  # passive slot
            heapq.heappush(ev, (t + 30, a, s))
            continue
        first.setdefault(a, t)
        seen[a] = t
        active = sum(1 for b, x in seen.items() if x > t - 270 or held[b])
        share = max(1, min(2 + int((t - first[a]) / ramp_s), math.ceil((len(CELLS) - i + sum(held)) / active)))
        known[a] = share
        if held[a] >= min(slots, share):
            r = refus.get((a, s), 0)
            refus[(a, s)] = r + 1
            heapq.heappush(ev, (t + min(300, 30 * 2 ** r), a, s))
            continue
        refus[(a, s)] = 0
        d = REF[CELLS[i]] * rng.uniform(0.9, 1.1)
        i += 1
        held[a] += 1
        e = t + d + OVER_S
        heapq.heappush(fin, (e, a))
        ends.append(e)
        heapq.heappush(ev, (e, a, s))
        claimed = t
    ends.sort()
    n = len(ends)
    return {"claimed": claimed / 60, "half": ends[n // 2 - 1] / 60, "p90": ends[int(0.9 * n) - 1] / 60, "full": ends[-1] / 60}


if __name__ == "__main__":
    for n in (40, 100, 250, 500):
        for p in (1, 8):
            runs = [sim(n, p, s) for s in range(20)]
            m = {k: statistics.mean(r[k] for r in runs) for k in runs[0]}
            print(f"{n:3d} attendees x {p} slots: all claimed {m['claimed']:5.1f} min, half {m['half']:5.1f}, 90% {m['p90']:5.1f}, full {m['full']:5.1f}")
