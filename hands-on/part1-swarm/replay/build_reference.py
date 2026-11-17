"""Build replay/reference.json: the public Harness Leaderboard's GLM-5.3-Flash TB2.1 trials
(effort low, the public board's scored set) for the grid's harnesses and tasks, as swarm
Result records. One-off data build, run with the harness-leaderboard virtualenv:

    cd ~/Documents/Code/harness-leaderboard
    .venv/bin/python <repo>/hands-on/part1-swarm/replay/build_reference.py \
        --inputs <board-inputs.json snapshot> --grid <repo>/hands-on/part1-swarm/config/grid.json

The snapshot is `bucket.board_snapshot` output (all runs' raw trial facts, run.json, invalid.json).
The board's own selection (`scoring.board_facts(public=True)`) decides which trials count, so the
reference matches what the public board shows. Cost = metered tokens x the run's list price.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hlb import scoring


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", default=str(Path(__file__).with_name("reference.json")))
    a = ap.parse_args(argv)
    grid = json.loads(Path(a.grid).read_text())
    snap = json.loads(Path(a.inputs).read_text())
    board = scoring.board_facts(snap["facts"], snap["runs"], snap["invalid"], public=True)
    tasks, harnesses = set(grid["tasks"]), set(grid["harnesses"])
    out = []
    base = datetime(2026, 11, 1, tzinfo=UTC)
    for (model, stage, effort, harness), (facts, price) in sorted(board.items()):
        if model != grid["model"] or stage != "full" or effort != grid["reasoning_effort"] or harness not in harnesses:
            continue
        per_task: dict[str, int] = {}
        for f in sorted(facts, key=lambda f: (f["task"], f["_order"], f["id"])):
            if f["benchmark"] != grid["benchmark"] or f["task"] not in tasks or f["_outcome"] not in ("pass", "fail"):
                continue
            trial = per_task.get(f["task"], 0)
            per_task[f["task"]] = trial + 1
            m = f.get("meter") or {}
            wall = (f.get("time_s") or {}).get("trial") or 0.0
            out.append({
                "cell_id": f"{harness}__{f['task']}__{trial}",
                "harness": harness, "harness_version": f.get("harness_version") or "",
                "task": f["task"], "trial": trial, "model": model,
                "reward": float(f.get("reward") or 0.0), "passed": f["_outcome"] == "pass",
                "started_at": base.isoformat(), "finished_at": (base + timedelta(seconds=wall)).isoformat(),
                "wall_s": round(wall, 1),
                "n_input_tokens": int(m.get("input") or 0), "n_cached_tokens": m.get("cached"),
                "n_output_tokens": int(m.get("output") or 0), "n_calls": m.get("n_calls"),
                "cost_usd": round(scoring.cost_usd(f, price), 6), "error": None,
                "hf_user": "harness-leaderboard", "runner_version": "reference",
                "sandbox_id": None, "trajectory_url": None, "source_trial_id": f["id"], "source_run": f["run_id"],
            })
    doc = {"source": "Harness Leaderboard public board, scoring.board_facts(public=True)",
           "snapshot_loaded_at": snap.get("loaded_at"), "model": grid["model"], "effort": grid["reasoning_effort"],
           "benchmark": grid["benchmark"], "n": len(out), "results": out}
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {a.out}: {len(out)} reference trials from snapshot {snap.get('loaded_at')}")


if __name__ == "__main__":
    main()
