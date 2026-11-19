"""Empty the STAGING board after a dry run: every file in its results bucket, then a restart (the Space rebuilds its
SQLite from the bucket, so it comes back empty). The prod Space and bucket are refused outright; there is no flag for them.

    HF_TOKEN=<org token> uv run python -m swarm_coordinator.reset            # empty, restart, check both boards read 0
    HF_TOKEN=<org token> uv run python -m swarm_coordinator.reset --pause    # empty and leave staging paused (no billing)

Order: pause first (a running Space flushes leases every 5 s and would write them back), delete in batches, list
the bucket again and refuse to go on while anything is left, then restart and read /api/state and /api/evolve/state.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from typing import Any

from . import deploy

BATCH = 500


def _files(api: Any, bucket: str) -> list[str]:
    return [e.path for e in api.list_bucket_tree(bucket, recursive=True) if getattr(e, "type", "file") == "file"]


def _wait(api: Any, space: str, want: str, sleep: Callable[[float], None], tries: int = 60) -> str:
    stage = ""
    for _ in range(tries):
        stage = api.get_space_runtime(space).stage
        if stage == want:
            return stage
        sleep(5)
    raise SystemExit(f"{space} did not reach {want} (stage {stage})")


def reset(api: Any, *, board: Callable[[], dict[str, int]], sleep: Callable[[float], None] = time.sleep, pause: bool = False,
          space: str | None = None, bucket: str | None = None) -> dict[str, Any]:
    """`board()` reads the restarted Space: {"done": Part 1 results, "evolve_runs": Part 2 runs}."""
    space, bucket = space or deploy.TARGETS["staging"][0], bucket or deploy.TARGETS["staging"][1]
    if space in deploy.TARGETS["prod"] or bucket in deploy.TARGETS["prod"]:
        raise SystemExit(f"refusing to reset prod ({space}, {bucket}): reset only empties the staging board")
    api.pause_space(space)
    _wait(api, space, "PAUSED", sleep)
    files = _files(api, bucket)
    for i in range(0, len(files), BATCH):
        api.batch_bucket_files(bucket, delete=files[i:i + BATCH])
    left = _files(api, bucket)
    if left:
        raise SystemExit(f"{len(left)} files still in {bucket} after the delete (e.g. {left[0]}); Space left paused")
    if pause:
        return {"deleted": len(files), "board": None, "stage": "PAUSED"}
    api.restart_space(space)
    stage = _wait(api, space, "RUNNING", sleep)
    for _ in range(30):  # RUNNING comes a few seconds before the app answers
        try:
            state = board()
            break
        except Exception:  # noqa: BLE001
            sleep(5)
    else:
        raise SystemExit(f"{deploy.space_url(space)} does not answer after the restart")
    if any(state.values()):
        raise SystemExit(f"the staging board is not empty after the restart: {state}")
    return {"deleted": len(files), "board": state, "stage": stage}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["staging"], default="staging")
    ap.add_argument("--pause", action="store_true", help="leave staging paused after emptying it")
    a = ap.parse_args(argv)
    import httpx
    from huggingface_hub import HfApi

    url = deploy.space_url(deploy.TARGETS[a.target][0])

    def board() -> dict[str, int]:
        with httpx.Client(base_url=url, timeout=30) as http:
            return {"done": http.get("/api/state").raise_for_status().json()["counts"]["done"],
                    "evolve_runs": http.get("/api/evolve/state").raise_for_status().json()["totals"]["runs"]}

    out = reset(HfApi(token=os.environ.get("HF_TOKEN")), board=board, pause=a.pause)
    print(f"staging reset: {out['deleted']} files deleted, Space {out['stage']}" + (f", board {out['board']}" if out["board"] else "") + f" ({url})")


if __name__ == "__main__":
    main()
