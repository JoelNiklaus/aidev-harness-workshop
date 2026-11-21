"""Room load test on STAGING with one account: N HF Jobs stand in for N attendees.

    HF_TOKEN=... uv run python scripts/room_loadtest.py part1 --n 40 --tag p1a        # 40 runner Jobs x 8 parallel trials
    HF_TOKEN=... uv run python scripts/room_loadtest.py part2 --n 40 --tag p2a        # 40 evolve loops, scripted proposer
    HF_TOKEN=... uv run python scripts/room_loadtest.py status --tag p1a              # Job stages
    HF_TOKEN=... uv run python scripts/room_loadtest.py collect --tag p1a --out x.jsonl   # the LT lines of every Job log
    HF_TOKEN=... uv run python scripts/room_loadtest.py cancel --tag p1a

Each Job installs the PUBLISHED wheel an attendee gets and runs its real entry point (`aidev-swarm run --loop --parallel 8`
or `aidev-evolve run --block ...`) through a small driver that changes three things only: the coordinator sees the Job
under its own load-test name (`lt-<tag>-NN`, header X-Aidev-Loadtest-As; the staging Space must be deployed with
`--loadtest-users <account>`), the GLM provider is hashed from that name (so the room's 3:1 provider split applies), and
measurements are printed as `LT {json}` lines. Part 2's coding agent is a script named `claude` that submits the
recorded code-audit patch after a fixed wait, so no subscription is used. Submissions are spread over `--arrive-s`
like a room that launches within a few minutes. The token goes only into each Job's secret HF_TOKEN.

What one account cannot show: per-account limits of fresh attendee accounts (each attendee has their own), and the HF
edge's per-IP limit towards the venue. Empty staging afterwards: `python -m swarm_coordinator.reset`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from swarm_coordinator import deploy, join, paths  # noqa: E402
from swarm_coordinator.app import LOADTEST_HEADER, LOADTEST_NAME  # noqa: E402

IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"  # the image both tools' Jobs use
LABEL = "aidev-loadtest"
MARK = "LT "
STAGING = deploy.space_url(deploy.TARGETS["staging"][0])
TIMEOUT_S = {"part1": 75 * 60, "part2": 50 * 60}

# ---- the drivers (run inside each Job, next to the installed wheel) ---------------------------------------------------

DRIVER_PART1 = r'''
import json, os, statistics, sys, time
from collections import Counter
from pathlib import Path
from aidev_swarm import cli, providers
from aidev_swarm.coordinator import Coordinator

NAME, HEADER = os.environ["LT_NAME"], os.environ["LT_HEADER"]
T0 = time.time()

def lt(**kw):
    print("LT " + json.dumps({"name": NAME, "t": round(time.time() - T0, 1), **kw}), flush=True)

_order = providers.order
providers.order = lambda user, provs: _order(NAME, provs)  # the provider of this simulated attendee

def coordinator(url, token):
    c = Coordinator(url, token)
    c._http.headers[HEADER] = NAME
    return c

def meter(trial_dir):
    p = Path(trial_dir) / "agent" / "hlb-meter.jsonl"
    calls = []
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "status" in d:
            calls.append(d)
    ttfb = [c["ttfb_s"] for c in calls if isinstance(c.get("ttfb_s"), (int, float))]
    return {"calls": len(calls), "status": dict(Counter(str(c.get("status")) for c in calls)),
            "served_by": dict(Counter(str(c.get("served_by")) for c in calls if c.get("status") == 200)),
            "routes": dict(Counter(str(c.get("model")) for c in calls)),
            "fallback": sum(1 for c in calls if c.get("fallback") or c.get("fallback_from")),
            "ttfb_p50": round(statistics.median(ttfb), 2) if ttfb else None, "ttfb_max": round(max(ttfb), 1) if ttfb else None,
            "latency_s": round(sum(c.get("latency_s") or 0 for c in calls), 1), "billed_usd": round(sum(c.get("billed_usd") or 0 for c in calls), 5)}

def run_trial(claim, grid, token, user, workdir):
    start = time.time()
    try:
        d = cli._run_trial(claim, grid, token, user, workdir)
    except BaseException as e:
        lt(k="trial_error", cell=claim["cell_id"], provider=grid.get("provider"), s=round(time.time() - start, 1), error=f"{type(e).__name__}: {str(e)[:300]}")
        raise
    try:
        res = json.loads((Path(d) / "result.json").read_text())
        exc = res.get("exception_info") or {}
        phases = {}
        for ph in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
            v = res.get(ph) or {}
            if v.get("started_at") and v.get("finished_at"):
                from datetime import datetime
                p = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
                phases[ph] = round((p(v["finished_at"]) - p(v["started_at"])).total_seconds(), 1)
        reward = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        lt(k="trial", cell=claim["cell_id"], harness=claim["harness"], task=claim["task"], provider=grid.get("provider"),
           s=round(time.time() - start, 1), reward=reward, exc=exc.get("exception_type"), exc_msg=(exc.get("exception_message") or "")[-200:],
           phases=phases, **meter(d))
    except Exception as e:
        lt(k="trial_unread", cell=claim["cell_id"], error=f"{type(e).__name__}: {str(e)[:200]}")
    return d

lt(k="start", provider=providers.order("x", providers.load())[0], parallel=int(os.environ["LT_PARALLEL"]))
rc = cli.main(["run", "--loop", "--parallel", os.environ["LT_PARALLEL"], "--board", os.environ["LT_BOARD"]],
              deps=cli.Deps(coordinator=coordinator, run_trial=run_trial))
lt(k="end", rc=rc)
sys.exit(0)
'''

DRIVER_PART2 = r'''
import json, os, re, stat, sys, time
from importlib import resources
from pathlib import Path
from aidev_evolve import board, cli, providers

NAME, HEADER = os.environ["LT_NAME"], os.environ["LT_HEADER"]
T0 = time.time()

def lt(**kw):
    print("LT " + json.dumps({"name": NAME, "t": round(time.time() - T0, 1), **kw}), flush=True)

# the scripted coding agent: answers the probe, then submits the recorded code-audit patch after a fixed wait
patch = Path(str(resources.files("aidev_evolve.data").joinpath("recorded", "candidates", "c001")))
shim = Path("/tmp/lt-bin/claude")
shim.parent.mkdir(parents=True, exist_ok=True)
shim.write_text(f"""#!{sys.executable}
import json, shutil, sys, time
prompt = sys.argv[sys.argv.index("-p") + 1]
if prompt.startswith("Reply with the single word"):
    print(json.dumps({{"result": "ok", "is_error": False}}))
else:
    time.sleep({float(os.environ.get("LT_PROPOSER_S", "45"))})
    for name in ("agent.py", "PATCH.md"):
        shutil.copy2("{patch}/" + name, name)
    print(json.dumps({{"result": "done", "is_error": False, "total_cost_usd": 0.0}}))
""")
shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
os.environ["PATH"] = f"{shim.parent}:{os.environ['PATH']}"

providers.for_user = lambda user: providers.order(NAME, providers.load())  # the provider of this simulated attendee
_init = board.Board.__init__
def init(self, url, token, **kw):
    _init(self, url, token, **kw)
    self._client.headers[HEADER] = NAME
board.Board.__init__ = init

run_dir = Path("/tmp/runs") / NAME
lt(k="start", provider=providers.for_user("x")[0], block=os.environ["LT_BLOCK"])
try:
    rc = cli.main(["run", "--block", os.environ["LT_BLOCK"], "--proposer", "claude", "--run-dir", str(run_dir), "--board", os.environ["LT_BOARD"]])
except BaseException as e:
    rc = f"{type(e).__name__}: {str(e)[:300]}"
lt(k="end", rc=rc)
try:
    state = json.loads((run_dir / "state.json").read_text())
    hist = [{k: h.get(k) for k in ("iteration", "verdict", "val_parent", "val_child", "iter_s", "iter_usd", "provider_calls", "detail")} for h in state["history"]]
    ho = state.get("heldout") or {}
    lt(k="run", incumbent=state["incumbent"], spend_usd=state["spend_usd"], history=hist, curve=state["curve"],
       heldout={k: ho.get(k) for k in ("seed", "best", "diff", "provider_calls")}, outbox=len((run_dir / "outbox.jsonl").read_text().splitlines()) if (run_dir / "outbox.jsonl").exists() else 0)
    for line in (run_dir / "transcript.jsonl").read_text().splitlines():
        ev = json.loads(line)
        m = re.search(r"Job (\w+): (\d+) s, \$([\d.]+)", ev["text"])
        if m:
            lt(k="job", at=ev["t"], job=m.group(1), seconds=int(m.group(2)), usd=float(m.group(3)))
        if "warning" in ev["text"] or "unreachable" in ev["text"] or "queued" in ev["text"] or "failed" in ev["text"]:
            lt(k="note", at=ev["t"], text=ev["text"].strip()[:300])
    for line in (run_dir / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        tr = r.get("trajectory") or {}
        lt(k="rec", eval=r["eval"], cand=r["candidate"], task=r["task"], outcome=r["outcome"], wall_s=r.get("wall_s"), agent_s=r.get("agent_s"),
           error=(r.get("error") or "")[:200] or None, provider_calls=r.get("provider_calls"), calls=(r.get("usage") or {}).get("calls"),
           end=tr.get("end"), cost_usd=r.get("cost_usd"))
except Exception as e:
    lt(k="collect_error", error=f"{type(e).__name__}: {str(e)[:300]}")
sys.exit(0)
'''


# ---- Job specs (pure) -------------------------------------------------------------------------------------------------

def names(tag: str, n: int) -> list[str]:
    out = [f"lt-{tag}-{i:02d}" for i in range(n)]
    bad = [x for x in out if not LOADTEST_NAME.match(x)]
    if bad:
        raise SystemExit(f"tag {tag!r} gives a name the coordinator refuses ({bad[0]}): lowercase letters, digits and dashes only")
    return out


def wheel_urls() -> dict[str, str]:
    """The published wheels attendees install: the `<WHEEL_URL>` lines of the two READMEs (what /join and /evolve/join show)."""
    return {"part1": join.wheel_url((paths.runner_dir() / "README.md").read_text()),
            "part2": join.wheel_url((paths.evolve_dir() / "README.md").read_text())}


def job_spec(kind: str, name: str, *, token: str, user: str, board: str, wheel: str, tag: str, parallel: int = 8, block: str = "right",
             proposer_s: float = 45.0) -> dict[str, Any]:
    """Keyword arguments for HfApi.run_job. The token appears only under `secrets`."""
    package, driver = {"part1": ("aidev-swarm[run]", DRIVER_PART1), "part2": ("aidev-evolve", DRIVER_PART2)}[kind]
    file = "/tmp/" + wheel.rsplit("/", 1)[-1]
    script = (f"set -e; curl -fsSL -o {shlex.quote(file)} {shlex.quote(wheel)}; echo \"$LT_DRIVER\" | base64 -d > /tmp/lt_driver.py; "
              f'exec uvx --python 3.12 --from "{package} @ file://{file}" python /tmp/lt_driver.py')
    env = {"LT_NAME": name, "LT_HEADER": LOADTEST_HEADER, "LT_BOARD": board, "LT_DRIVER": base64.b64encode(driver.encode()).decode(),
           "LT_PARALLEL": str(parallel), "LT_BLOCK": block, "LT_PROPOSER_S": str(proposer_s),
           "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    return {"image": IMAGE, "command": ["bash", "-c", script], "env": env, "secrets": {"HF_TOKEN": token}, "flavor": "cpu-basic",
            "timeout": TIMEOUT_S[kind], "labels": {LABEL: tag, "aidev-loadtest-kind": kind}, "namespace": user}


def plan(kind: str, tag: str, n: int, *, arrive_s: float, **kw: Any) -> list[tuple[float, dict[str, Any]]]:
    """(seconds after the start, Job spec) per simulated attendee: arrivals spread evenly, seating blocks in turn."""
    blocks = list(join.BLOCKS)
    return [(arrive_s * i / max(1, n - 1) if n > 1 else 0.0, job_spec(kind, name, tag=tag, block=blocks[i % len(blocks)], **kw))
            for i, name in enumerate(names(tag, n))]


def lt_lines(log_lines: Iterator[str]) -> list[dict[str, Any]]:
    out = []
    for line in log_lines:
        i = line.find(MARK + "{")
        if i < 0:
            continue
        try:
            out.append(json.loads(line[i + len(MARK):]))
        except ValueError:
            continue
    return out


# ---- side effects -----------------------------------------------------------------------------------------------------

def submit(api: Any, jobs: list[tuple[float, dict[str, Any]]], *, sleep: Callable[[float], None] = time.sleep,
           clock: Callable[[], float] = time.monotonic, say: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Submit at each Job's offset; a refused submission is recorded and the next one still goes out."""
    start, out = clock(), []
    for at, spec in jobs:
        wait = at - (clock() - start)
        if wait > 0:
            sleep(wait)
        name = spec["env"]["LT_NAME"]
        try:
            job = api.run_job(**spec)
            out.append({"name": name, "job": job.id, "at": round(clock() - start, 1)})
        except Exception as e:  # noqa: BLE001 - the refusal is a measurement
            out.append({"name": name, "job": None, "at": round(clock() - start, 1), "error": f"{type(e).__name__}: {str(e)[:300]}"})
            say(f"{name}: not started ({out[-1]['error'][:160]})")
    return out


def tagged_jobs(api: Any, user: str, tag: str) -> list[Any]:
    return list(api.list_jobs(namespace=user, labels={LABEL: tag}))


def stage_of(job: Any) -> str:
    return str(job.status.stage).upper().split(".")[-1]


def job_log(token: str, user: str, job_id: str, *, budget_s: float = 40.0) -> list[str]:
    """The Job's log lines so far, bounded in wall time (a live stream never ends on its own)."""
    import httpx

    lines, deadline = [], time.monotonic() + budget_s
    try:
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(10.0, read=8.0)) as http:
            with http.stream("GET", f"https://huggingface.co/api/jobs/{user}/{job_id}/logs", headers={"Authorization": f"Bearer {token}"}) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if raw.startswith("data:"):
                        try:
                            lines.append(json.loads(raw[5:]).get("data", ""))
                        except ValueError:
                            pass
                    if time.monotonic() > deadline:
                        break
    except httpx.HTTPError:
        pass  # a read timeout on a quiet live stream: what was read so far is the log
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for kind in ("part1", "part2"):
        p = sub.add_parser(kind)
        p.add_argument("--n", type=int, default=40)
        p.add_argument("--tag", required=True)
        p.add_argument("--board", default=STAGING)
        p.add_argument("--arrive-s", type=float, default=180.0, help="spread the launches over this long")
        p.add_argument("--parallel", type=int, default=8)
        p.add_argument("--proposer-s", type=float, default=45.0)
    for name in ("status", "collect", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--out")
        p.add_argument("--skip", type=int, default=0, help="collect: start at this Job (to continue in a second call)")
        p.add_argument("--budget-s", type=float, default=330.0)
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    if a.cmd in ("part1", "part2"):
        if a.board.rstrip("/") == deploy.space_url(deploy.DEFAULT_SPACE):
            raise SystemExit("refusing to load-test the workshop board: use the staging Space")
        done = submit(api, plan(a.cmd, a.tag, a.n, arrive_s=a.arrive_s, token=token, user=user, board=a.board, wheel=wheel_urls()[a.cmd],
                                parallel=a.parallel, proposer_s=a.proposer_s))
        print(json.dumps({"started": sum(1 for d in done if d["job"]), "refused": [d for d in done if not d["job"]], "tag": a.tag}))
        return 0
    jobs = tagged_jobs(api, user, a.tag)
    if a.cmd == "status":
        stages: dict[str, int] = {}
        for j in jobs:
            stages[stage_of(j)] = stages.get(stage_of(j), 0) + 1
        print(json.dumps({"jobs": len(jobs), "stages": stages}))
    elif a.cmd == "cancel":
        for j in jobs:
            if stage_of(j) in ("RUNNING", "SCHEDULING", "PENDING", "QUEUED", "STARTING"):
                api.cancel_job(job_id=j.id, namespace=user)
        print(f"cancelled the live Jobs of {len(jobs)}")
    else:
        end, n = time.monotonic() + a.budget_s, 0
        jobs = sorted(jobs, key=lambda j: j.id)
        with open(a.out, "a") as f:
            for i, j in enumerate(jobs):
                if i < a.skip:
                    continue
                if time.monotonic() > end:
                    print(f"time is up: continue with --skip {i}")
                    break
                rows = lt_lines(iter(job_log(token, user, j.id)))
                for row in rows:
                    f.write(json.dumps({"job": j.id, "stage": stage_of(j), **row}) + "\n")
                n += len(rows)
            else:
                print(f"all {len(jobs)} Jobs read")
        print(f"{n} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
