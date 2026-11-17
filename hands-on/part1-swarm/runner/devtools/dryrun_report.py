"""Summarize a swarm dry run of launched HF Jobs: start latency, trial wall time, failures by cause, cost.

    uv run python devtools/dryrun_report.py --launches launches.log --state state.json [--out report.json]

`launches.log` holds the `launch` output lines (Job ids are read from "launched HF Job <id>"); `state.json` is
the coordinator's /api/state with full results (the stand-in's), or a list of Result records. Reads each
runner Job (inspect + logs) and each trial's sandbox Job (by the Result's `sandbox_id`) with the local HF login.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

LINE = re.compile(r"^\[aidev-swarm (\d\d):(\d\d):(\d\d)\] (.*)$")
CLAIMED = re.compile(r"^claimed (\S+) on (\S+) \(trial (\d+)\)")
VERDICT = re.compile(r"^(PASS|FAIL) (\S+) (\S+):")
# $/h by flavor (`hf jobs hardware`, read 2026-11-17); an exposed port adds $0.01/h (runner PROGRESS, 2026-11-17)
PRICE_H = {"cpu-basic": 0.01, "cpu-upgrade": 0.03}
EXPOSED_PORT_H = 0.01


def parse_log(text: str, day: datetime) -> list[dict[str, Any]]:
    """Runner events from a Job log. Times are the container's (UTC) clock on `day`."""
    out = []
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        at = day.replace(hour=int(m[1]), minute=int(m[2]), second=int(m[3]), microsecond=0)
        msg = m[4]
        if c := CLAIMED.match(msg):
            out.append({"at": at, "kind": "claimed", "harness": c[1], "task": c[2], "trial": int(c[3])})
        elif v := VERDICT.match(msg):
            out.append({"at": at, "kind": v[1], "harness": v[2], "task": v[3]})
        elif msg.startswith("trial crashed ("):
            out.append({"at": at, "kind": "crash", "detail": msg[len("trial crashed ("):]})
        elif msg.startswith("infrastructure failure before any model call ("):
            out.append({"at": at, "kind": "infra", "detail": msg[len("infrastructure failure before any model call ("):]})
        elif msg.startswith("result not accepted"):
            out.append({"at": at, "kind": "not_accepted", "detail": msg})
        elif msg.startswith("nothing left to claim"):
            out.append({"at": at, "kind": "grid_full"})
        elif msg.startswith("stopping:"):
            out.append({"at": at, "kind": "stopped", "detail": msg})
    return out


def failure_causes(event_lists: list[list[dict[str, Any]]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for events in event_lists:
        for e in events:
            if e["kind"] in ("crash", "infra"):
                c[f"{e['kind']}: {e['detail'].split(':', 1)[0]}"] += 1
            elif e["kind"] in ("not_accepted", "stopped"):
                c[e["kind"]] += 1
    return dict(c)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def _secs(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() if a and b else None


def _cost(job: Any, end: datetime | None = None) -> float:
    """Job cost from its run time. A cancelled Job (Harbor cancels each sandbox) has no finished_at: bill it until `end`."""
    stop = job.finished_at or end
    if not (job.started_at and stop):
        return 0.0
    rate = PRICE_H[job.flavor] + (EXPOSED_PORT_H if getattr(job.status, "expose_urls", None) else 0.0)
    return round(_secs(job.started_at, stop) / 3600 * rate, 10)


def collect(api: Any, job_ids: list[str], results: list[dict[str, Any]]) -> dict[str, Any]:
    runners, events = [], []
    for jid in job_ids:
        j = api.inspect_job(job_id=jid)
        day = j.created_at.replace(hour=0, minute=0, second=0, microsecond=0)
        ev = parse_log("\n".join(api.fetch_job_logs(job_id=jid)), day)
        ev = [{**e, "at": e["at"] + timedelta(days=1) if e["at"] < j.created_at - timedelta(minutes=5) else e["at"]} for e in ev]  # past midnight
        events.append(ev)
        claims = [e for e in ev if e["kind"] == "claimed"]
        runners.append({"id": jid, "stage": j.status.stage, "created": j.created_at, "started": j.started_at, "finished": j.finished_at,
                        "first_claim": claims[0]["at"] if claims else None, "trials": len(claims), "cost": _cost(j)})
    by_cell = {r["cell_id"]: r for r in results}
    sandboxes = []
    for r in by_cell.values():
        if r.get("sandbox_id"):
            s = api.inspect_job(job_id=r["sandbox_id"])
            end = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00")) if r.get("finished_at") else None
            sandboxes.append({"cell_id": r["cell_id"], "created": s.created_at, "started": s.started_at, "finished": s.finished_at or end,
                              "cost": _cost(s, end), "flavor": s.flavor})
    first_sandbox = {}
    for run, ev in zip(runners, events):
        for e in ev:
            if e["kind"] == "claimed":
                cid = f"{e['harness']}__{e['task']}__{e['trial']}"
                sb = next((s for s in sandboxes if s["cell_id"] == cid), None)
                if sb and sb["started"]:
                    first_sandbox[run["id"]] = sb["started"]
                    break
    walls = [r["wall_s"] for r in results if r.get("wall_s") is not None]
    sb_lat = []
    for run, ev in zip(runners, events):
        for e in ev:
            if e["kind"] == "claimed":
                sb = next((s for s in sandboxes if s["cell_id"] == f"{e['harness']}__{e['task']}__{e['trial']}"), None)
                if sb and sb["started"]:
                    sb_lat.append(_secs(e["at"], sb["started"]))
    def dist(xs):
        xs = [x for x in xs if x is not None]
        return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "max": max(xs) if xs else None}
    return {
        "runner_jobs": len(runners), "runner_stages": dict(Counter(r["stage"] for r in runners)),
        "launch_to_job_running_s": dist([_secs(r["created"], r["started"]) for r in runners]),
        "launch_to_first_claim_s": dist([_secs(r["created"], r["first_claim"]) for r in runners]),
        "launch_to_first_sandbox_running_s": dist([_secs(r["created"], first_sandbox.get(r["id"])) for r in runners]),
        "claim_to_sandbox_running_s": dist(sb_lat),
        "trial_wall_s": dist(walls),
        "results": len(results), "passed": sum(1 for r in results if r.get("passed")),
        "results_by_harness": {h: {"n": sum(1 for r in results if r["harness"] == h), "passed": sum(1 for r in results if r["harness"] == h and r.get("passed"))}
                               for h in sorted({r["harness"] for r in results})},
        "results_with_error": dict(Counter((r.get("error") or "").split(":", 1)[0] for r in results if r.get("error"))),
        "failures_by_cause": failure_causes(events),
        "trials_claimed": sum(r["trials"] for r in runners),
        "cost_usd": {"inference_list_price": round(sum(r.get("cost_usd") or 0 for r in results), 4),
                     "sandbox_jobs": round(sum(s["cost"] for s in sandboxes), 4),
                     "runner_jobs": round(sum(r["cost"] for r in runners), 4)},
        "runner_job_minutes": dist([(_secs(r["started"], r["finished"]) or 0) / 60 for r in runners]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launches", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ids = re.findall(r"launched HF Job ([0-9a-f]{24})", Path(a.launches).read_text())
    state = json.loads(Path(a.state).read_text())
    results = state["results"] if isinstance(state, dict) else state
    from huggingface_hub import HfApi

    rep = collect(HfApi(), ids, results)
    text = json.dumps(rep, indent=2, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
