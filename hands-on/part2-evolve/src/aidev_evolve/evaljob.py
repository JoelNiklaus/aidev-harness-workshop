"""What runs inside the HF Job: Harbor trials of each candidate on HF Sandboxes, results streamed back.

The laptop sends a spec (candidates' sources, which tasks and how many trials each) as one environment
variable. Each candidate becomes one Harbor job with the Part 1 sandbox setup (the Harness Leaderboard's
`HLBSandboxEnvironment`, same pin) and our `EvolveAgent`; all jobs run at once. A watcher thread emits
each finished trial as a record into the Job log (transport.py), so the laptop shows progress live; the
last line says how many records were sent.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from aidev_evolve import transport
from aidev_evolve.providers import MODEL, REFERENCE

BENCH = "terminal-bench-2.1"
EFFORT = "low"
AGENT_IMPORT = "aidev_evolve.harbor_agent:EvolveAgent"
HARBOR_SLACK_S = 90  # Harbor's own agent timeout sits this far behind the runtime budget, so the runtime ends first
SANDBOX_USD_PER_H = 0.02  # cpu-basic $0.01/h plus the exposed port $0.01/h (Part 1 runner PROGRESS)
SPEC_ENV = "AIDEV_EVOLVE_SPEC"
# The tasks' own verifier timeouts are 6-60 min; on the board their p90 run time is at most 3 min on the search
# and validation tasks (4 min on held-out pytorch-model-cli). A hung test run would otherwise hold the whole
# iteration (seen 17 Nov: one headless-terminal verifier kept a Job open for 10+ min). A timeout is not scored.
VERIFIER_CAP_S = 240


def encode_spec(spec: dict[str, Any]) -> str:
    return base64.b64encode(gzip.compress(json.dumps(spec).encode())).decode()


def decode_spec(text: str) -> dict[str, Any]:
    return json.loads(gzip.decompress(base64.b64decode(text)))


def spec_providers(spec: dict[str, Any]) -> list[str]:
    """The attendee's GLM providers, preferred first (providers.order on the laptop); the reference when a spec names
    none (devtools sweeps)."""
    return list(spec.get("providers") or [REFERENCE])


def build_run(model: str = MODEL, effort: str = EFFORT, provider: str = REFERENCE) -> dict[str, Any]:
    """The run record hlb's Harbor config is built from (as in Part 1's runner trial.build_run)."""
    from hlb import config as hlb_config

    return {"run_id": f"evolve-{uuid.uuid4().hex[:8]}", "model": model, "provider": provider, "reasoning_effort": effort,
            "benchmark_setups": {b: hlb_config.benchmark_setup(b) for b in hlb_config.benchmarks()},
            "route": hlb_config.route(model, provider), "model_limits": hlb_config.model_limits(model, provider),
            "meter_settings": hlb_config.meter_settings(model, provider), "npm_before": {}}


def prepare_env(run: dict[str, Any], environ: dict[str, str] | None = None) -> None:
    """Token for sandboxes and for the (unused) in-sandbox meter; the attendee's namespace for sandbox lookups
    (Part 1 runner trial.prepare_env)."""
    from hlb import config as hlb_config
    from hlb.worker import hand_out_sandbox_key
    from huggingface_hub import whoami

    environ = os.environ if environ is None else environ
    token = environ["HF_TOKEN"]
    environ["HF_INFERENCE_TOKEN"] = token
    hand_out_sandbox_key(run["run_id"], environ)
    hlb_config.OWNER = whoami(token=token)["name"]


def harbor_configs(spec: dict[str, Any], jobs_dir: Path, run_id: str, run: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    from hlb.worker import harbor_job_config

    run = run or {**build_run(), "run_id": run_id}
    budget = float(spec.get("budget_s", 150))
    providers = spec_providers(spec)
    cfgs = []
    for i, entry in enumerate(spec["runs"]):
        cand, tasks, trials = entry["candidate"], entry["tasks"], int(entry.get("trials", 1))
        job_name = f"{run_id}-{i}-{cand}"  # a candidate can appear in two runs (seed: search x1, validation x2)
        cfg = harbor_job_config(run, BENCH, ["mini-swe-agent"], tasks, trials, job_name, Path(jobs_dir),
                                {"aidev-evolve": run_id}, concurrency=len(tasks) * trials)
        cfg["agents"] = [{"import_path": AGENT_IMPORT, "model_name": run["model"], "override_timeout_sec": budget + HARBOR_SLACK_S,
                          "kwargs": {"source": spec["candidates"][cand], "candidate": cand, "budget_s": budget, "providers": providers}}]
        cfg["verifier"] = {"override_timeout_sec": VERIFIER_CAP_S}
        cfgs.append(cfg)
    return cfgs


def _seconds(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    p = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
    return round((p(b) - p(a)).total_seconds(), 3)


def record(trial_dir: Path, candidate: str) -> dict[str, Any]:
    trial_dir = Path(trial_dir)
    result = json.loads((trial_dir / "result.json").read_text())
    tpath = trial_dir / "agent" / "evolve-trajectory.json"
    traj = json.loads(tpath.read_text()) if tpath.exists() else None
    verdict = result.get("verifier_result")
    exc = result.get("exception_info") or {}
    reward = float(((verdict or {}).get("rewards") or {}).get("reward", 0.0))
    if (traj and traj.get("end") == "infra") or (verdict is None and exc.get("exception_type") not in (None, "AgentTimeoutError")):
        outcome = "infra"
    else:
        outcome = "pass" if reward >= 1.0 else "fail"
    wall = _seconds(result.get("started_at"), result.get("finished_at"))
    inference = float((traj or {}).get("cost_usd") or 0.0)
    sandbox = (wall or 0.0) / 3600 * SANDBOX_USD_PER_H
    ae = result.get("agent_execution") or {}
    return {"candidate": candidate, "task": result["task_name"].rsplit("/", 1)[-1], "trial_name": result.get("trial_name", trial_dir.name),
            "outcome": outcome, "reward": reward, "wall_s": wall, "agent_s": _seconds(ae.get("started_at"), ae.get("finished_at")),
            "error": (f"{exc.get('exception_type')}: {(exc.get('exception_message') or '')[-300:]}" if exc else None),
            "usage": (traj or {}).get("usage"), "inference_usd": round(inference, 6), "sandbox_usd": round(sandbox, 6),
            "cost_usd": round(inference + sandbox, 6), "provider_calls": (traj or {}).get("provider_calls") or {}, "trajectory": traj}


def harbor_run(cfgs: list[dict[str, Any]]) -> None:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    async def go() -> None:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=256))  # sandbox execs run in threads
        jobs = [await Job.create(JobConfig.model_validate(c)) for c in cfgs]
        for cfg, res in zip(cfgs, await asyncio.gather(*(j.run() for j in jobs), return_exceptions=True)):
            if isinstance(res, BaseException):
                print(f"aidev-evolve: Harbor job {cfg['job_name']} failed: {type(res).__name__}: {res}", flush=True)

    asyncio.run(go())


def main(spec: dict[str, Any], *, workdir: Path, harbor: Callable[[list[dict[str, Any]]], None] = harbor_run,
         emit: Callable[[str], None] = print, prepare: Callable[[dict[str, Any]], None] = prepare_env, poll_s: float = 2.0) -> int:
    run = build_run(provider=spec_providers(spec)[0])
    prepare(run)
    cfgs = harbor_configs(spec, Path(workdir), run["run_id"], run)
    cand_of = {c["job_name"]: c["agents"][0]["kwargs"]["candidate"] for c in cfgs}
    seen: set[Path] = set()
    lock = threading.Lock()
    stop = threading.Event()

    def sweep() -> None:
        with lock:
            for job_name, cand in cand_of.items():
                for res in sorted((Path(workdir) / job_name).glob("*/result.json")):
                    d = res.parent
                    if d in seen:
                        continue
                    try:
                        if not json.loads(res.read_text()).get("finished_at"):
                            continue
                        rec = record(d, cand)
                    except (json.JSONDecodeError, OSError):
                        continue  # still being written
                    seen.add(d)
                    for line in transport.encode(rec, msg_id=f"{cand}-{d.name}"):
                        emit(line)

    def watch() -> None:
        while not stop.wait(poll_s):
            sweep()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        harbor(cfgs)
    except Exception as e:  # noqa: BLE001 - report it; the laptop counts the missing trials
        emit(f"aidev-evolve: Harbor failed: {type(e).__name__}: {e}")
    finally:
        stop.set()
        watcher.join()
        sweep()
        emit(transport.done_line(len(seen)))
    return len(seen)


def cli() -> int:
    """Entry point inside the Job: `aidev-evolve eval-job` reads the spec from $AIDEV_EVOLVE_SPEC."""
    spec = decode_spec(os.environ[SPEC_ENV])
    start = time.monotonic()
    n = main(spec, workdir=Path("/tmp/evolve-jobs"), emit=lambda s: print(s, flush=True))
    print(f"aidev-evolve: {n} trials in {time.monotonic() - start:.0f} s", flush=True)
    return 0
