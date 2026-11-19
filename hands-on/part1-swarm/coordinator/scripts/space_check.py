"""Live check of the deployed coordinator Space, end to end through its real storage.

    HF_TOKEN=<a token that can read the private Space and its bucket> uv run python scripts/space_check.py [--target staging]

1. /healthz answers; /api/state reads 0 of all cells done and the full reference layer.
2. One claim -> submit of a SYNTHETIC result (error field says so) through the API; the result file must appear in
   the results bucket (the Space flushes every 5 s).
3. The file is deleted from the bucket, the Space restarted, and /api/state must read 0 done again.
Exits non-zero at the first failed step. The token is only sent to huggingface.co and the Space; never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_coordinator import deploy, paths  # noqa: E402
from swarm_coordinator.app import MIN_RUNNER  # noqa: E402
from swarm_coordinator.grid import Grid  # noqa: E402

USER_AGENT = f"aidev-swarm/{'.'.join(map(str, MIN_RUNNER))} (space-check)"  # the coordinator takes claims only from current runners


def synthetic_result(claim: dict[str, Any], user: str) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    return {"cell_id": claim["cell_id"], "harness": claim["harness"], "harness_version": "check", "task": claim["task"],
            "trial": claim["trial"], "model": claim["model"], "reward": 0.0, "passed": False, "started_at": now,
            "finished_at": now, "wall_s": 0.0, "n_input_tokens": 0, "n_cached_tokens": 0, "n_output_tokens": 0,
            "n_calls": 0, "cost_usd": 0.0, "error": "synthetic result from scripts/space_check.py (deleted right after)",
            "hf_user": user, "runner_version": "space_check", "sandbox_id": None, "trajectory_url": None}


def state_problems(state: dict[str, Any], *, done: int, grid: Grid) -> list[str]:
    c = state["counts"]
    problems = []
    if c["done"] != done:
        problems.append(f"done is {c['done']}, expected {done}")
    cells = len(grid.harnesses) * len(grid.tasks) * grid.trials_per_cell
    if c["cells_total"] != cells:
        problems.append(f"cells_total is {c['cells_total']}, expected {cells}")
    want_ref = len(grid.tasks) * grid.trials_per_cell
    for h in state["harnesses"]:
        n = (h.get("reference") or {}).get("n")
        if n != want_ref:
            problems.append(f"reference for {h['harness']} has n={n}, expected {want_ref}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    deploy.add_target_args(ap)
    return deploy.resolve_target(ap.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    import httpx
    from huggingface_hub import HfApi

    a = parse_args(argv)
    token = os.environ["HF_TOKEN"]
    api, url, grid = HfApi(token=token), deploy.space_url(a.space), Grid.load(paths.grid_path())
    http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}, timeout=60)
    user = api.whoami()["name"]

    def wait_running() -> None:
        for _ in range(90):
            if api.get_space_runtime(a.space).stage == "RUNNING":
                for _ in range(30):
                    try:
                        if http.get("/healthz").status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(5)
            time.sleep(10)
        raise SystemExit(f"{a.space} is not RUNNING with /healthz 200: {api.get_space_runtime(a.space).stage}")

    def check_state(done: int, step: str) -> None:
        problems = state_problems(http.get("/api/state").json(), done=done, grid=grid)
        print(f"{step}: /api/state {'ok' if not problems else 'FAIL ' + '; '.join(problems)}")
        if problems:
            raise SystemExit(1)

    wait_running()
    print("1. /healthz 200 at", url)
    check_state(0, "1.")
    claim = http.post("/api/claim")
    claim.raise_for_status()
    cell = claim.json()
    cell = cell[0] if isinstance(cell, list) else cell
    r = http.post(f"/api/submit/{cell['cell_id']}", json=synthetic_result(cell, user))
    r.raise_for_status()
    path = f"results/{cell['cell_id']}.json"
    for _ in range(24):
        if any(e.path == path for e in api.list_bucket_tree(a.bucket, prefix="results/", recursive=True)):
            break
        time.sleep(5)
    else:
        raise SystemExit(f"2. FAIL: {path} did not reach {a.bucket}")
    print(f"2. claim {cell['cell_id']} -> submit 200 -> {a.bucket}/{path} present")
    check_state(1, "2.")
    api.batch_bucket_files(a.bucket, delete=[path])
    left = [e.path for e in api.list_bucket_tree(a.bucket, prefix="results/", recursive=True)]
    if left:
        raise SystemExit(f"3. FAIL: results left in the bucket: {left}")
    api.restart_space(a.space)
    time.sleep(20)
    wait_running()
    check_state(0, "3. deleted and restarted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
