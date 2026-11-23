"""Measure the seed agent once and write the scores the wheel ships (`src/aidev_evolve/data/seed_calibration.json`).

    uv run python devtools/seed_calibration.py launch                 # one HF Job, detached; its id goes to runs/p5/
    uv run python devtools/seed_calibration.py launch --top-up        # one more search trial per task (after an infra trial)
    uv run python devtools/seed_calibration.py collect                # when the Jobs are done: read their logs, write the file

Run it again whenever the seed, the splits, the time budget or the evaluation pin changes (tests/test_workshop_run.py
refuses a calibration of another seed or budget). The Job is the attendee's evaluation Job of the published wheel: seed
on the search tasks x SETS and on the validation tasks x VAL_TRIALS, on the reference provider. `calibration.py` says
what the file is for. Dev tool (not shipped in the wheel).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from aidev_evolve import __version__, calibration, launch, loop, transport

SETS, VAL_TRIALS = 4, 8
HERE = Path(__file__).resolve().parents[1]
JOB_FILE = HERE / "runs" / "p5" / "seed-calibration-job.json"
OUT = HERE / "src" / "aidev_evolve" / "data" / calibration.FILE


def spec(splits: dict[str, Any], top_up: bool = False) -> dict[str, Any]:
    runs = [{"candidate": "c000", "tasks": splits["search"], "trials": 1 if top_up else SETS}]
    if not top_up:
        runs.append({"candidate": "c000", "tasks": splits["validation"], "trials": VAL_TRIALS})
    return {"candidates": {"c000": loop.SEED_SOURCE}, "budget_s": splits["budget_s"], "runs": runs}


def decode(lines: Any) -> tuple[list[dict[str, Any]], bool]:
    dec, records, done = transport.Decoder(), [], False
    for line in lines:
        records += dec.feed(line)
        done = done or transport.is_done(line)
    return records, done


def log_lines(job: str, user: str, token: str, seconds: float = 120.0) -> list[str]:
    """The Job's log so far, read for at most `seconds` (the log API streams; a finished Job's stream ends by itself)."""
    import httpx

    out, end = [], time.monotonic() + seconds
    with httpx.Client(trust_env=False, timeout=30) as c, c.stream("GET", f"https://huggingface.co/api/jobs/{user}/{job}/logs",
                                                                 headers={"Authorization": f"Bearer {token}"}) as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]).get("data", ""))
            if time.monotonic() > end:
                break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["launch", "collect"])
    ap.add_argument("--top-up", action="store_true", help="launch: add one search trial per task to the Jobs already there")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, get_token

    token = get_token()
    api = HfApi(token=token)
    splits = loop.default_splits()
    jobs = json.loads(JOB_FILE.read_text()) if JOB_FILE.exists() and a.top_up or a.cmd == "collect" else []
    if a.cmd == "launch":
        user = api.whoami()["name"]
        job = api.run_job(**launch.job_spec(token, user, spec(splits, a.top_up)))
        JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOB_FILE.write_text(json.dumps([*jobs, {"job": job.id, "user": user, "version": __version__}]))
        print(f"Job {job.id} submitted; `collect` when it is done")
        return 0
    records: list[dict[str, Any]] = []
    for j in jobs:
        recs, done = decode(log_lines(j["job"], j["user"], token))
        if not done:
            print(f"Job {j['job']} has sent {len(recs)} records and is not done yet")
            return 1
        records += recs
    cal = calibration.build(records, splits, seed_source=loop.SEED_SOURCE, budget_s=splits["budget_s"], sets=SETS,
                            measured={"jobs": [j["job"] for j in jobs], "version": jobs[0]["version"], "date": time.strftime("%Y-%m-%d"), "provider": "zai-org",
                                      "trials": len(records), "infra": sum(r["outcome"] == "infra" for r in records)})
    OUT.write_text(json.dumps(cal, separators=(",", ":")) + "\n")
    print(f"{OUT}: {len(cal['search_sets'])} search sets, {len(cal['validation'])} validation trials, {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
