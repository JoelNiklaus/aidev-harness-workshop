"""Live check of /evolve on the deployed Space, through the Part 2 client and the Space's real storage.

    HF_TOKEN=<a token that can read the private Space and write its bucket> uv run python scripts/evolve_space_check.py [--target staging] [--keep]

1. Posts a short real loop (the recorded Phase 4 run baseline-1: seed, iterations 1-2, held-out with its strategy) under the
   run id `check-<time>` with `aidev_evolve.board.Board` (the attendee CLI's own client, from ../../part2-evolve).
2. /api/evolve/state must show the run (curve, kept patch in the feed, held-out, its strategy in the ranking) next to
   the recorded backdrop;
   /api/evolve/diff serves the kept patch; the four event files must reach the bucket (flushed every 5 s).
3. Unless --keep: deletes the files, restarts the Space, and /api/evolve/state must no longer show the run.
Exits non-zero at the first failed step. The token goes only to huggingface.co and the Space; never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[3] / "part2-evolve" / "src"))

from swarm_coordinator import deploy, evolve, paths  # noqa: E402
from swarm_coordinator.evolve_replay import load_recorded  # noqa: E402


def check_events(recorded: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    """baseline-1's seed, iterations 1-2 and held-out, as the loop's `_submit` sends them (no schema: Board adds it)."""
    events = next(r for r in recorded if r["name"] == "baseline-1")["events"]
    keep = [e for e in events if e["kind"] in ("seed", "heldout") or (e["kind"] == "iteration" and e["iteration"] <= 2)]
    return [{**{k: v for k, v in e.items() if k not in ("t", "schema")}, "run_id": run_id} for e in keep]


def problems(state: dict[str, Any], user: str, run_id: str) -> list[str]:
    out = []
    run = next((r for r in state["runs"] if r["user"] == user and r["run_id"] == run_id), None)
    if run is None:
        out.append(f"run {run_id} of {user} is not on the board")
    else:
        xs = sorted({c[0] for c in run["curve"]})  # a step line: a kept patch's iteration appears twice
        if xs != [0, 1, 2]:
            out.append(f"curve has iterations {xs}, expected [0, 1, 2]")
        if run["heldout"] is None:
            out.append("held-out result missing")
        if not any(f["run_id"] == run_id for f in state["feed"]):
            out.append("kept patch missing from the feed")
    if not state["reference"]:
        out.append("no recorded backdrop")
    if not any(run_id in e["runs"] for e in state.get("presets", {}).get("strategies", [])):
        out.append(f"{run_id} missing from the strategy ranking")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    deploy.add_target_args(ap)
    ap.add_argument("--keep", action="store_true", help="leave the check run on the board (for screenshots); delete it later with a rerun")
    return deploy.resolve_target(ap.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    import httpx
    from aidev_evolve.board import Board
    from huggingface_hub import HfApi

    a = parse_args(argv)
    token = os.environ["HF_TOKEN"]
    api, url = HfApi(token=token), deploy.space_url(a.space)
    http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    user = api.whoami()["name"]
    run_id = f"check-{int(time.time())}"

    board = Board(url, token)
    for e in check_events(load_recorded(paths.evolve_replay_path()), run_id):
        board.submit(e)
    print(f"1. posted seed, iterations 1-2, held-out as {user}/{run_id} through aidev_evolve.board.Board")

    time.sleep(2.5)  # the state is cached 2 s
    state = http.get("/api/evolve/state").json()
    bad = problems(state, user, run_id)
    diff = http.get(f"/api/evolve/diff/{user}/{run_id}/1")
    if diff.status_code != 200 or not diff.text.startswith("---"):
        bad.append(f"diff endpoint answered {diff.status_code}")
    if bad:
        raise SystemExit("2. FAIL: " + "; ".join(bad))
    prefix = f"{evolve.SINK_PREFIX}{user}/{run_id}/"
    for _ in range(24):
        files = [e.path for e in api.list_bucket_tree(a.bucket, prefix=prefix, recursive=True)]
        if len(files) == 4:
            break
        time.sleep(5)
    else:
        raise SystemExit(f"2. FAIL: {len(files)} of 4 event files reached {a.bucket}/{prefix}")
    print(f"2. /api/evolve/state shows the run, feed and held-out; diff served; 4 files in {a.bucket}/{prefix}")
    if a.keep:
        print(f"   kept on the board (--keep): rerun without --keep, or delete {prefix} and restart the Space")
        return 0

    api.batch_bucket_files(a.bucket, delete=files)
    api.restart_space(a.space)
    time.sleep(20)
    for _ in range(60):
        try:
            if http.get("/healthz").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    state = http.get("/api/evolve/state").json()
    if any(r["run_id"] == run_id for r in state["runs"]):
        raise SystemExit("3. FAIL: the run is still on the board after delete + restart")
    print(f"3. deleted and restarted: {state['totals']['runs']} runs on the board")
    return 0


if __name__ == "__main__":
    sys.exit(main())
