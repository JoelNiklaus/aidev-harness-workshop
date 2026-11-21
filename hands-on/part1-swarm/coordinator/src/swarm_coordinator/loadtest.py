"""Load test: N async fake runners claim, heartbeat and submit against a coordinator until the grid is full.

    uv run swarm-loadtest --runners 300            # starts a local coordinator (--fake-auth) and hammers it
    uv run swarm-loadtest --url http://host:port --runners 300

Reports latency per endpoint (p50/p95/max), errors, and whether any cell was handed out twice
(client side: every claim response logged per cell; server side: the store's max claims per cell).

    uv run swarm-loadtest --runners 500 --trial-s 60 180 --kill-frac 0.1

Runners "run" each trial for a random time in --trial-s while heartbeating every --heartbeat-s (30 s, as the
runner); --kill-frac of them die mid-trial without releasing (a killed Job). The report gives the seconds from
each death to the next claim of its cell, and checks that no cell was handed out twice while its holder lived
and that no cell counted two results.

    uv run swarm-loadtest --evolve --attendees 500 --interval 30 --iterations 6

Part 2: N attendees' evolve loops post a seed event, then one iteration event (with a 4 KB diff) every
`interval` seconds, starts spread over the first interval; viewers poll /api/evolve/state every 3 s.
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

from .app import MIN_RUNNER

HEADERS = {"User-Agent": f"aidev-swarm/{'.'.join(map(str, MIN_RUNNER))} (loadtest)"}  # the coordinator takes claims only from current runners


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def latency_summary(lat: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {k: {"n": len(v), "p50": round(percentile(v, 50), 1), "p95": round(percentile(v, 95), 1), "max": round(max(v), 1)}
            for k, v in sorted(lat.items()) if v}


def fake_result(cell: dict[str, Any], user: str, i: int) -> dict[str, Any]:
    return {"cell_id": cell["cell_id"], "harness": cell["harness"], "harness_version": "loadtest", "task": cell["task"],
            "trial": cell["trial"], "model": cell["model"], "reward": float(i % 2), "passed": i % 2 == 1,
            "started_at": "2026-11-30T15:00:00+00:00", "finished_at": "2026-11-30T15:05:00+00:00", "wall_s": 300.0,
            "n_input_tokens": 400_000 + i, "n_cached_tokens": 200_000, "n_output_tokens": 8_000, "n_calls": 25,
            "cost_usd": 0.0, "error": None, "hf_user": user, "runner_version": "loadtest", "sandbox_id": None, "trajectory_url": None}


def _doomed(n: int, kill_frac: float) -> set[int]:
    """The runners that die mid-trial: round(kill_frac * n) of them, spread evenly over the list."""
    k = round(n * kill_frac)
    return {round(j * n / k) for j in range(k)} if k else set()


async def _drive(base_url: str, users: list[str], *, transport: httpx.AsyncBaseTransport | None = None, viewers: int = 0,
                 timeout_s: float = 60.0, trial_s: tuple[float, float] = (0.0, 0.0), heartbeat_s: float = 30.0,
                 kill: set[int] | None = None, wait_s: float | None = None, seed: int = 0, parallel: int = 1) -> dict[str, Any]:
    """Run the given fake runners (and viewers) to the end of the grid; raw latencies, claims, submits, deaths and errors.

    A runner claims, "runs" its trial for a random time in `trial_s` while heartbeating every `heartbeat_s`, and
    submits. The runners whose index is in `kill` die once, partway through their first trial: no release, no more
    heartbeats (a SIGKILLed Job). A 204 with Retry-After (leased cells may come back) waits `wait_s` (default: the
    header) and claims again; a plain 204 (grid full) ends the runner. With `parallel` > 1 each runner is an attendee
    whose Job runs that many slots (aidev-swarm --parallel), each claiming with limit=parallel; a 409 at the attendee's
    limit or share is a wait (`wait_s`, default 30 s), not an error."""
    import random

    rng = random.Random(seed)
    lat: dict[str, list[float]] = defaultdict(list)
    claims: dict[str, list[list[Any]]] = defaultdict(list)  # cell -> [[user, unix time], ...]
    submits: dict[str, int] = defaultdict(int)  # cell -> accepted submits
    deaths: dict[str, list[Any]] = {}  # cell -> [user, unix time of death]
    errors: list[str] = []
    held: dict[str, int] = defaultdict(int)  # user -> cells its slots hold now
    max_held = [0]
    stop = asyncio.Event()
    kill = kill or set()
    conns = len(users) * parallel + viewers + 10
    limits = httpx.Limits(max_connections=conns, max_keepalive_connections=conns)
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s, limits=limits, headers=HEADERS) as client:

        async def call(kind: str, path: str, user: str, **kw: Any) -> httpx.Response | None:
            t0 = time.perf_counter()
            try:
                r = await client.post(path, headers={"Authorization": f"Bearer {user}"}, **kw)
            except httpx.HTTPError as e:
                errors.append(f"{kind}: {type(e).__name__}")
                return None
            lat[kind].append((time.perf_counter() - t0) * 1000)
            if r.status_code >= 400 and not (kind == "claim" and r.status_code == 409 and parallel > 1):
                errors.append(f"{kind}: HTTP {r.status_code} {r.text[:100]}")
            return r

        async def runner(i: int, user: str) -> None:
            room = {"share": 2, "full": False}  # what the runner's slots share (aidev-swarm cli.Session.share / .full)
            await asyncio.gather(*(slot(i, user, k, room) for k in range(parallel)))

        async def slot(i: int, user: str, k: int, room: dict[str, Any]) -> None:
            await asyncio.sleep((i % 50) * 0.002 + k * 0.001)  # a small ramp, as attendees do not start in the same ms
            doomed = i in kill and k == 0
            params = {"limit": parallel} if parallel > 1 else None
            refusals = 0

            def backoff(base: float) -> float:  # the runner's slot_wait: a refused slot asks less and less often
                return wait_s if wait_s is not None else (min(300.0, base * 2 ** refusals) if parallel > 1 else base)

            for j in range(10_000):
                if parallel > 1 and room["full"]:
                    return
                if parallel > 1 and k > room["share"]:  # the runner's passive slots: the slot at the share asks
                    await asyncio.sleep(wait_s if wait_s is not None else 30.0)
                    continue
                r = await call("claim", "/api/claim", user, params=params)
                if r is not None and r.status_code == 409 and parallel > 1:
                    room["share"] = r.json().get("share") or room["share"]
                    await asyncio.sleep(backoff(30.0))
                    refusals += 1
                    continue
                if r is None or r.status_code >= 400:
                    return
                if r.status_code == 204:
                    if "retry-after" not in r.headers:
                        room["full"] = True
                        return
                    await asyncio.sleep(backoff(float(r.headers["retry-after"])))
                    refusals += 1
                    continue
                refusals = 0
                room["share"] = r.json().get("share", room["share"])
                cell = r.json()
                cid = cell["cell_id"]
                claims[cid].append([user, time.time()])
                held[user] += 1
                max_held[0] = max(max_held[0], held[user])
                dur = rng.uniform(*trial_s)
                die_after = dur * rng.uniform(0.2, 0.8) if doomed else None
                ran, beats = 0.0, 0
                while ran < (die_after if doomed else dur) or (not doomed and beats == 0):
                    step = min(heartbeat_s, (die_after if doomed else dur) - ran)
                    if step > 0:
                        await asyncio.sleep(step)
                        ran += step
                    if doomed and ran >= die_after:
                        break
                    await call("heartbeat", f"/api/heartbeat/{cid}", user)
                    beats += 1
                if doomed:
                    deaths[cid] = [user, time.time()]
                    return
                r = await call("submit", f"/api/submit/{cid}", user, json=fake_result(cell, user, j))
                held[user] -= 1
                if r is not None and r.status_code == 200:
                    submits[cid] += 1

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
    return {"lat": dict(lat), "claims": dict(claims), "submits": dict(submits), "deaths": deaths, "errors": errors, "wall": wall,
            "max_held": max_held[0]}


def _drive_in_process(base_url: str, users: list[str], viewers: int, kw: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_drive(base_url, users, viewers=viewers, **kw))


def merge(raws: list[dict[str, Any]]) -> dict[str, Any]:
    lat: dict[str, list[float]] = defaultdict(list)
    claims: dict[str, list[list[Any]]] = defaultdict(list)
    submits: dict[str, int] = defaultdict(int)
    deaths: dict[str, list[Any]] = {}
    errors: list[str] = []
    for r in raws:
        for k, v in r["lat"].items():
            lat[k] += v
        for c, us in r["claims"].items():
            claims[c] += us
        for c, n in r["submits"].items():
            submits[c] += n
        deaths.update(r["deaths"])
        errors += r["errors"]
    for us in claims.values():
        us.sort(key=lambda u: u[1])
    return {"lat": dict(lat), "claims": dict(claims), "submits": dict(submits), "deaths": deaths, "errors": errors,
            "wall": max((r["wall"] for r in raws), default=0.0), "max_held": max((r.get("max_held", 0) for r in raws), default=0)}


def reclaims(claims: dict[str, list[list[Any]]], deaths: dict[str, list[Any]]) -> tuple[list[float], list[str]]:
    """Seconds from each death to the next claim of its cell, and the cells claimed twice without a death in between."""
    waits, unexpected = [], []
    for cid, us in claims.items():
        us = sorted(us, key=lambda u: u[1])
        for (u1, _), (_, t2) in zip(us, us[1:]):
            dead = deaths.get(cid)
            if dead and dead[0] == u1:
                waits.append(t2 - dead[1])
            else:
                unexpected.append(cid)
    return waits, sorted(set(unexpected))


async def run_loadtest(base_url: str, n_runners: int = 300, *, transport: httpx.AsyncBaseTransport | None = None, viewers: int = 0,
                       procs: int = 1, trial_s: tuple[float, float] = (0.0, 0.0), heartbeat_s: float = 30.0, kill_frac: float = 0.0,
                       wait_s: float | None = None, parallel: int = 1) -> dict[str, Any]:
    """procs > 1 spreads the runners over processes, so the load generator is not the bottleneck
    (one Python process with hundreds of connections saturates its own core first)."""
    users = [f"loadtest-{i:04d}" for i in range(n_runners)]
    doomed = _doomed(n_runners, kill_frac)
    if procs <= 1:
        raw = await _drive(base_url, users, transport=transport, viewers=viewers, trial_s=trial_s, heartbeat_s=heartbeat_s,
                           kill=doomed, wait_s=wait_s, parallel=parallel)
    else:
        from concurrent.futures import ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(procs) as ex:
            idx = [list(range(n_runners))[i::procs] for i in range(procs)]
            raws = await asyncio.gather(*(loop.run_in_executor(
                ex, _drive_in_process, base_url, [users[k] for k in part], viewers // procs + (i < viewers % procs),
                {"trial_s": trial_s, "heartbeat_s": heartbeat_s, "wait_s": wait_s, "seed": i, "parallel": parallel,
                 "kill": {pos for pos, k in enumerate(part) if k in doomed}})
                for i, part in enumerate(idx)))
        raw = merge(list(raws))
    lat, claims, errors, wall = raw["lat"], raw["claims"], raw["errors"], raw["wall"]
    waits, unexpected = reclaims(claims, raw["deaths"])
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=60) as client:
        # /api/state is cached for 2 s: ask until it reflects the last submit (or give up after ~4 s)
        submitted = sum(raw["submits"].values())
        for _ in range(8):
            final = (await client.get("/api/state")).json()
            if final["counts"]["done"] >= submitted:
                break
            await asyncio.sleep(0.5)
    sent = sum(len(v) for k, v in lat.items() if k != "state")
    return {
        "runners": n_runners, "parallel": parallel, "max_leases_per_user": raw["max_held"], "viewers": viewers, "procs": procs, "trial_s": list(trial_s), "heartbeat_s": heartbeat_s,
        "wall_s": round(wall, 2), "requests": sum(len(v) for v in lat.values()),
        "rps": round(sent / wall, 1) if wall else None,
        "heartbeat_rps": round(len(lat.get("heartbeat", [])) / wall, 1) if wall else None,
        "latency_ms": latency_summary(lat),
        "errors": len(errors), "error_samples": errors[:5],
        "killed": len(raw["deaths"]), "reclaimed": len(waits),
        "reclaim_s": {"p50": round(percentile(waits, 50), 2), "max": round(max(waits), 2)} if waits else None,
        "cells_claimed": len(claims), "unexpected_reclaims": unexpected,
        "double_counted": sorted(c for c, n in raw["submits"].items() if n > 1),
        "cells_total": final["counts"]["cells_total"], "cells_done": final["counts"]["done"],
        "server_max_claims_per_cell": final["counts"]["max_claims_per_cell"],
    }


PRESETS = ("baseline", "failures-first", "small-diffs", "two-candidates", "every-weakness", "code-audit")
HELDOUT_TASKS = [f"heldout-{t:02d}" for t in range(23)]


def fake_strategy(k: int) -> dict[str, Any]:
    preset = PRESETS[k % len(PRESETS)]
    return {"name": preset, "preset": preset, "hash": f"{k % len(PRESETS):010x}", "modified": False, "prompt": "Read the traces first. " * 40,
            "knobs": {"traces": "failed", "max_traces": 4, "history": "titles", "diffs": False, "candidates": 1, "val_trials": 2, "z": 1.5,
                      "confirm": True, "iterations": 4}}


def fake_evolve_event(user: str, i: int, *, k: int = 0, heldout: bool = False) -> dict[str, Any]:
    """Attendee k's i-th event: seed (0), iterations, or the held-out check with per-task counts; each with a strategy."""
    run_id = f"{user}-run"
    base = {"schema": "aidev-evolve/1", "run_id": run_id, "strategy": fake_strategy(k), "provider": ("novita", "zai-org")[k % 2]}
    if heldout:
        return {**base, "kind": "heldout", "seed": 0.6, "best": 0.8, "best_candidate": "c001", "n_tasks": 23, "trials": 2,
                "per_task": {t: [(k + j) % 3, 2, (k + 2 * j) % 3, 2] for j, t in enumerate(HELDOUT_TASKS)}, "kept": ["Run until time budget"],
                "spend_usd": 1.0}
    if i == 0:
        return {**base, "kind": "seed", "val_score": 0.4, "search_score": 0.0, "spend_usd": 0.11}
    diff = "--- agent.py\n+++ agent.py\n" + "".join(f"+line {n} of a patch that changes the harness loop\n" for n in range(80))
    return {**base, "kind": "iteration", "iteration": i, "candidate": f"c{i:03d}", "parent": "c000",
            "title": ["Run until time budget, not 10 steps", "Show stderr to the model", "Verify before finishing"][i % 3],
            "hypothesis": "Hypothesis: runs stop at the step cap with most of the budget left.", "accepted": i == 1,
            "verdict": "kept" if i == 1 else "dropped", "val_child": 0.85, "val_parent": 0.4, "best_val": 0.85,
            "fixed": ["build-pmars"], "broke": [], "diff": diff, "spend_usd": 0.11 + 0.15 * i}


async def run_evolve_loadtest(base_url: str, attendees: int = 500, *, interval_s: float = 30.0, iterations: int = 6, viewers: int = 20,
                              transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 60.0) -> dict[str, Any]:
    lat: dict[str, list[float]] = defaultdict(list)
    errors: list[str] = []
    limited = 0
    stop = asyncio.Event()
    limits = httpx.Limits(max_connections=attendees + viewers + 10, max_keepalive_connections=attendees + viewers + 10)
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s, limits=limits) as client:

        async def attendee(k: int) -> None:
            nonlocal limited
            user = f"evolve-{k:04d}"
            await asyncio.sleep(interval_s * k / max(attendees, 1))  # starts spread over one interval
            for i in range(iterations + 2):  # seed, iterations, held-out
                t0 = time.perf_counter()
                event = fake_evolve_event(user, i, k=k, heldout=i == iterations + 1)
                try:
                    r = await client.post("/api/evolve/submit", json=event, headers={"Authorization": f"Bearer {user}"})
                    lat["submit"].append((time.perf_counter() - t0) * 1000)
                    if r.status_code == 429:
                        limited += 1
                    elif r.status_code >= 400:
                        errors.append(f"submit: HTTP {r.status_code} {r.text[:100]}")
                except httpx.HTTPError as e:
                    errors.append(f"submit: {type(e).__name__}")
                if i <= iterations:
                    await asyncio.sleep(interval_s)

        async def viewer() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    r = await client.get("/api/evolve/state")
                    lat["state"].append((time.perf_counter() - t0) * 1000)
                    if r.status_code >= 400:
                        errors.append(f"state: HTTP {r.status_code}")
                except httpx.HTTPError as e:
                    errors.append(f"state: {type(e).__name__}")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=min(3.0, interval_s))
                except TimeoutError:
                    pass

        t0 = time.perf_counter()
        views = [asyncio.create_task(viewer()) for _ in range(viewers)]
        await asyncio.gather(*(attendee(k) for k in range(attendees)))
        wall = time.perf_counter() - t0
        stop.set()
        await asyncio.gather(*views)
        want = attendees * iterations
        for _ in range(120):  # /api/evolve/state is cached 2 s; a big ranking waits 10x its cost before it reranks
            st = (await client.get("/api/evolve/state")).json()
            board = st["totals"]
            if board["iterations"] >= want and st["presets"]["n_runs"] >= attendees:
                break
            await asyncio.sleep(0.5)
    return {"attendees": attendees, "interval_s": interval_s, "iterations": iterations, "viewers": viewers, "wall_s": round(wall, 1),
            "submit_rps": round(len(lat["submit"]) / wall, 1) if wall else None,
            "latency_ms": latency_summary(lat),
            "errors": len(errors), "error_samples": errors[:5], "rate_limited": limited, "board": board,
            "ranked_runs": st["presets"]["n_runs"], "presets": len(st["presets"]["strategies"])}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="swarm-loadtest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="an already running coordinator started with --fake-auth (default: start one locally)")
    ap.add_argument("--runners", type=int, default=300)
    ap.add_argument("--viewers", type=int, default=20)
    ap.add_argument("--trial-s", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"), help="fake trial length range")
    ap.add_argument("--heartbeat-s", type=float, default=30.0, help="runner heartbeat interval (the runner's HEARTBEAT_S)")
    ap.add_argument("--parallel", type=int, default=1, help="slots per runner (aidev-swarm --parallel), each claiming with limit=N")
    ap.add_argument("--kill-frac", type=float, default=0.0, help="share of runners that die mid-trial without releasing")
    ap.add_argument("--lease-s", type=float, help="lease of the locally started coordinator (default: the store's)")
    ap.add_argument("--procs", type=int, default=6, help="load-generator processes")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--evolve", action="store_true", help="Part 2: evolve submissions instead of claims")
    ap.add_argument("--attendees", type=int, default=500)
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between one attendee's events")
    ap.add_argument("--iterations", type=int, default=6)
    a = ap.parse_args(argv)
    proc = None
    url = a.url
    if not url:
        port = _free_port()
        env = {k: v for k, v in os.environ.items() if k not in ("SWARM_BUCKET", "SPACE_ID")}
        lease = ["--lease-s", str(a.lease_s)] if a.lease_s else []
        proc = subprocess.Popen([sys.executable, "-m", "swarm_coordinator.cli", "--fake-auth", "--port", str(port), *lease], env=env)
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(url + "/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
    try:
        if a.evolve:
            report = asyncio.run(run_evolve_loadtest(url, a.attendees, interval_s=a.interval, iterations=a.iterations, viewers=a.viewers))
        else:
            report = asyncio.run(run_loadtest(url, a.runners, viewers=a.viewers, procs=a.procs, trial_s=tuple(a.trial_s),
                                              heartbeat_s=a.heartbeat_s, kill_frac=a.kill_frac, parallel=a.parallel))
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    if a.evolve:
        ok = report["errors"] == 0 and report["rate_limited"] == 0 and report["board"]["iterations"] == a.attendees * a.iterations
    else:
        ok = (not report["unexpected_reclaims"] and not report["double_counted"] and report["reclaimed"] == report["killed"]
              and report["errors"] == 0 and report["cells_done"] == report["cells_total"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
