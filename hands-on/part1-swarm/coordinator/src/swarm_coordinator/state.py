"""The board payload (GET /api/state): leaderboard per harness, grid fill, feed, counters.

Everything is computed from one store snapshot plus the fixed reference layer, so the page
needs no second request.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .grid import Grid

FEED_LEN = 30


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval of k passes in n trials."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _tokens(r: dict[str, Any]) -> int:
    return int(r.get("n_input_tokens") or 0) + int(r.get("n_output_tokens") or 0)


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by[r["harness"]].append(r)
    out = {}
    for h, rs in by.items():
        n = len(rs)
        k = sum(1 for r in rs if r["passed"])
        cost = sum(float(r.get("cost_usd") or 0.0) for r in rs)
        lo, hi = wilson(k, n)
        out[h] = {"n": n, "n_pass": k, "pass_rate": k / n, "ci": [lo, hi], "cost_usd": round(cost, 4),
                  "cost_per_trial": cost / n, "cost_per_pass": cost / k if k else None,
                  "tokens_per_trial": sum(_tokens(r) for r in rs) / n,
                  "wall_s_per_trial": sum(float(r.get("wall_s") or 0.0) for r in rs) / n}
    return out


def by_provider(results: list[dict[str, Any]]) -> dict[str, int]:
    """Results per GLM-5.3-Flash provider (PROTOCOL.md "Providers"); runners before 0.4.1 name none: "unknown"."""
    out: dict[str, int] = defaultdict(int)
    for r in results:
        out[r.get("provider") or "unknown"] += 1
    return dict(sorted(out.items()))


def build_state(grid: Grid, snap: dict[str, Any], reference: list[dict[str, Any]], meta: dict[str, dict[str, str]], *, mode: str) -> dict[str, Any]:
    in_grid = {c.cell_id for c in grid.cells}
    results = [r for r in snap["results"] if r["cell_id"] in in_grid or (r["harness"] in grid.harnesses and r["task"] in grid.tasks and r["trial"] >= grid.trials_per_cell)]
    status = {r["cell_id"]: ("p" if r["passed"] else "f") for r in results}
    running = {le["cell_id"] for le in snap["leases"]}
    rows = {}
    for h in grid.harnesses:
        rows[h] = "".join(status.get(f"{h}__{t}__{k}") or ("c" if f"{h}__{t}__{k}" in running else "o")
                          for t in grid.tasks for k in range(grid.trials_per_cell))
    room, ref = aggregate(results), aggregate([r for r in reference if r["harness"] in grid.harnesses and r["task"] in grid.tasks])
    reached_by_claim = {le["cell_id"].split("__")[0] for le in snap["leases"]}
    harnesses = [{"harness": h, "display": (meta.get(h) or {}).get("display", h), "color": (meta.get(h) or {}).get("color", "#888888"),
                  "reached": h in room or h in reached_by_claim, "room": room.get(h), "reference": ref.get(h)} for h in grid.harnesses]
    feed = [{"hf_user": r["hf_user"], "harness": r["harness"], "task": r["task"], "trial": r["trial"], "passed": r["passed"],
             "cost_usd": r.get("cost_usd"), "wall_s": r.get("wall_s"), "at": r.get("received_at")}
            for r in reversed(results[-FEED_LEN:])]
    contributors = {r["hf_user"] for r in results} | {le["user"] for le in snap["leases"]}
    return {
        "mode": mode, "model": grid.model, "generated_at": snap["now"], "version": snap["version"],
        "counts": {"cells_total": len(in_grid), "done": len(results), "base_done": sum(r["cell_id"] in in_grid for r in results),
                   "extra_done": sum(r["cell_id"] not in in_grid for r in results), "passed": sum(1 for r in results if r["passed"]),
                   "running": len(running), "contributors": len(contributors), "max_claims_per_cell": snap["max_claims"],
                   "by_provider": by_provider(results)},
        "harnesses": harnesses,
        "grid": {"tasks": list(grid.tasks), "trials": grid.trials_per_cell, "rows": rows},
        "feed": feed,
    }
