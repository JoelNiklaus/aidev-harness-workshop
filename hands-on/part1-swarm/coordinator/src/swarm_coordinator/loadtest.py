"""Load test: N async fake runners claim, heartbeat and submit against a coordinator until the grid is full.

    uv run swarm-loadtest --runners 300            # starts a local coordinator (--fake-auth) and hammers it
    uv run swarm-loadtest --url http://host:port --runners 300

Reports latency per endpoint (p50/p95/max), errors, and whether any cell was handed out twice
(client side: every claim response logged per cell; server side: the store's max claims per cell).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import httpx


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def fake_result(cell: dict[str, Any], user: str, i: int) -> dict[str, Any]:
    return {"cell_id": cell["cell_id"], "harness": cell["harness"], "harness_version": "loadtest", "task": cell["task"],
            "trial": cell["trial"], "model": cell["model"], "reward": float(i % 2), "passed": i % 2 == 1,
            "started_at": "2026-11-30T15:00:00+00:00", "finished_at": "2026-11-30T15:05:00+00:00", "wall_s": 300.0,
            "n_input_tokens": 400_000 + i, "n_cached_tokens": 200_000, "n_output_tokens": 8_000, "n_calls": 25,
            "cost_usd": 0.0, "error": None, "hf_user": user, "runner_version": "loadtest", "sandbox_id": None, "trajectory_url": None}


async def _drive(base_url: str, users: list[str], *, transport: httpx.AsyncBaseTransport | None = None, heartbeats: int = 1,
                 think_s: float = 0.0, viewers: int = 0, timeout_s: float = 60.0) -> dict[str, Any]:
    """Run the given fake runners (and viewers) to the end of the grid; raw latencies, claims and errors."""
    lat: dict[str, list[float]] = defaultdict(list)
    claims: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []
    stop = asyncio.Event()
    limits = httpx.Limits(max_connections=len(users) + viewers + 10, max_keepalive_connections=len(users) + viewers + 10)
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s, limits=limits) as client:

        async def call(kind: str, path: str, user: str, **kw: Any) -> httpx.Response | None:
            t0 = time.perf_counter()
            try:
                r = await client.post(path, headers={"Authorization": f"Bearer {user}"}, **kw)
            except httpx.HTTPError as e:
                errors.append(f"{kind}: {type(e).__name__}")
                return None
            lat[kind].append((time.perf_counter() - t0) * 1000)
            if r.status_code >= 400:
                errors.append(f"{kind}: HTTP {r.status_code} {r.text[:100]}")
            return r

        async def runner(i: int, user: str) -> None:
            await asyncio.sleep((i % 50) * 0.002)  # a small ramp, as attendees do not start in the same ms
            for j in range(10_000):
                r = await call("claim", "/api/claim", user)
                if r is None or r.status_code == 204 or r.status_code >= 400:
                    return
                cell = r.json()
                claims[cell["cell_id"]].append(user)
                for _ in range(heartbeats):
                    await asyncio.sleep(think_s)
                    await call("heartbeat", f"/api/heartbeat/{cell['cell_id']}", user)
                await asyncio.sleep(think_s)
                await call("submit", f"/api/submit/{cell['cell_id']}", user, json=fake_result(cell, user, j))

        async def viewer() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    r = await client.get("/api/state")
                    lat["state"].append((time.perf_counter() - t0) * 1000)
                    if r.status_code >= 400:
                        errors.append(f"state: HTTP {r.status_code}")
                except httpx.HTTPError as e:
                    errors.append(f"state: {type(e).__name__}")
                await asyncio.sleep(0.5)

        t0 = time.perf_counter()
        views = [asyncio.create_task(viewer()) for _ in range(viewers)]
        await asyncio.gather(*(runner(i, u) for i, u in enumerate(users)))
        wall = time.perf_counter() - t0
        stop.set()
        await asyncio.gather(*views)
    return {"lat": dict(lat), "claims": dict(claims), "errors": errors, "wall": wall}


def _drive_in_process(base_url: str, users: list[str], heartbeats: int, viewers: int) -> dict[str, Any]:
    return asyncio.run(_drive(base_url, users, heartbeats=heartbeats, viewers=viewers))


def merge(raws: list[dict[str, Any]]) -> dict[str, Any]:
    lat: dict[str, list[float]] = defaultdict(list)
    claims: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []
    for r in raws:
        for k, v in r["lat"].items():
            lat[k] += v
        for c, us in r["claims"].items():
            claims[c] += us
        errors += r["errors"]
    return {"lat": dict(lat), "claims": dict(claims), "errors": errors, "wall": max((r["wall"] for r in raws), default=0.0)}


async def run_loadtest(base_url: str, n_runners: int = 300, *, transport: httpx.AsyncBaseTransport | None = None,
                       heartbeats: int = 1, viewers: int = 0, procs: int = 1) -> dict[str, Any]:
    """procs > 1 spreads the runners over processes, so the load generator is not the bottleneck
    (one Python process with hundreds of connections saturates its own core first)."""
    users = [f"loadtest-{i:04d}" for i in range(n_runners)]
    if procs <= 1:
        raw = await _drive(base_url, users, transport=transport, heartbeats=heartbeats, viewers=viewers)
    else:
        from concurrent.futures import ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(procs) as ex:
            parts = [users[i::procs] for i in range(procs)]
            raws = await asyncio.gather(*(loop.run_in_executor(ex, _drive_in_process, base_url, part, heartbeats, viewers // procs + (i < viewers % procs))
                                          for i, part in enumerate(parts)))
        raw = merge(list(raws))
    lat, claims, errors, wall = raw["lat"], raw["claims"], raw["errors"], raw["wall"]
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=60) as client:
        # /api/state is cached for 2 s: ask until it reflects the last submit (or give up after ~4 s)
        submitted = len(lat.get("submit", []))
        for _ in range(8):
            final = (await client.get("/api/state")).json()
            if final["counts"]["done"] >= submitted:
                break
            await asyncio.sleep(0.5)
    return {
        "runners": n_runners, "viewers": viewers, "procs": procs, "wall_s": round(wall, 2),
        "requests": sum(len(v) for v in lat.values()),
        "rps": round(sum(len(v) for k, v in lat.items() if k != "state") / wall, 1) if wall else None,
        "latency_ms": {k: {"n": len(v), "p50": round(percentile(v, 50), 1), "p95": round(percentile(v, 95), 1), "max": round(max(v), 1)}
                       for k, v in sorted(lat.items())},
        "errors": len(errors), "error_samples": errors[:5],
        "cells_claimed": len(claims), "double_claimed": sorted(c for c, us in claims.items() if len(us) > 1),
        "cells_total": final["counts"]["cells_total"], "cells_done": final["counts"]["done"],
        "server_max_claims_per_cell": final["counts"]["max_claims_per_cell"],
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="swarm-loadtest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="an already running coordinator started with --fake-auth (default: start one locally)")
    ap.add_argument("--runners", type=int, default=300)
    ap.add_argument("--viewers", type=int, default=20)
    ap.add_argument("--heartbeats", type=int, default=1)
    ap.add_argument("--procs", type=int, default=6, help="load-generator processes")
    ap.add_argument("--out", help="write the report JSON here")
    a = ap.parse_args(argv)
    proc = None
    url = a.url
    if not url:
        port = _free_port()
        env = {k: v for k, v in os.environ.items() if k not in ("SWARM_BUCKET", "SPACE_ID")}
        proc = subprocess.Popen([sys.executable, "-m", "swarm_coordinator.cli", "--fake-auth", "--port", str(port)], env=env)
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(url + "/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
    try:
        report = asyncio.run(run_loadtest(url, a.runners, heartbeats=a.heartbeats, viewers=a.viewers, procs=a.procs))
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    ok = not report["double_claimed"] and report["server_max_claims_per_cell"] <= 1 and report["errors"] == 0 and report["cells_done"] == report["cells_total"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
