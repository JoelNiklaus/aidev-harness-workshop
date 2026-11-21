"""Run one swarm cell as one Harbor trial on HF Sandbox, under the attendee's own HF token.

Reuses the Harness Leaderboard's wiring unchanged (`hlb`, pinned to the deployed Space revision):
the per-harness Harbor agent configs, the HF Sandbox environment with the in-sandbox metering proxy,
and the key custody (harnesses get a per-run dummy key; only the meter holds the real token). This is
`scripts/local_proof.py` of that repository reduced to one (harness, task) and made namespace-aware.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from hlb import config as hlb_config
from hlb.worker import hand_out_sandbox_key, harbor_job_config

CELL_LABEL = "aidev-swarm-cell"
# One run per process, as a leaderboard worker: hlb hands the harnesses their dummy key through os.environ, which the
# parallel slots (threads) of one Job share, so every trial of the process must use the same run id and key.
RUN_ID = f"swarm-{uuid.uuid4().hex[:8]}"


class TrialError(RuntimeError):
    """Harbor ended without leaving a trial folder."""


def build_run(model: str, effort: str, provider: str) -> dict[str, Any]:
    """The run record Harbor's job config is built from, as the leaderboard's launcher records one:
    the model's reference route, plus a provider pool of the reference and `provider` when another
    admitted provider serves the trial (its meter rewrites the route model; harness configs stay the same)."""
    m = hlb_config.models()[model]
    ref = m["default_provider"]
    if provider not in hlb_config.pool_providers(model) and provider != ref:
        raise ValueError(f"provider {provider!r} is not admitted for {model} (admitted: {hlb_config.pool_providers(model)})")
    run = {"run_id": RUN_ID, "model": model, "provider": ref, "reasoning_effort": effort,
           "benchmark_setups": {b: hlb_config.benchmark_setup(b) for b in hlb_config.benchmarks()},
           "route": hlb_config.route(model, ref), "model_limits": hlb_config.model_limits(model, ref),
           "meter_settings": hlb_config.meter_settings(model, ref), "npm_before": {}}
    if provider != ref:
        run["providers"] = {p: {"route_model": hlb_config.route_model(model, p), "meter_settings": hlb_config.meter_settings(model, p)} for p in (ref, provider)}
    return run


def prepare_env(environ: MutableMapping[str, str], token: str, hf_user: str, run_id: str) -> None:
    """The worker environment hlb expects: the token for sandboxes (HF_TOKEN) and for the meter
    (HF_INFERENCE_TOKEN), the harnesses' dummy key, and the attendee's namespace, where hlb looks up
    and cancels its sandbox Jobs (hlb hardcodes the leaderboard owner there)."""
    environ["HF_TOKEN"] = token
    environ["HF_INFERENCE_TOKEN"] = token
    hand_out_sandbox_key(run_id, environ)
    hlb_config.OWNER = hf_user


def job_config(run: dict[str, Any], harness: str, task: str, cell_id: str, jobs_dir: Path, job_name: str, provider: str | None = None,
               benchmark: str = "terminal-bench-2.1") -> dict[str, Any]:
    unit_provider = provider if "providers" in run else None
    return harbor_job_config(run, benchmark, [harness], [task], 1, job_name, Path(jobs_dir), {CELL_LABEL: hlb_config.label_value(cell_id)},
                             concurrency=1, provider=unit_provider)


def harbor_run(cfg: dict[str, Any]) -> None:
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    async def go() -> None:
        await (await Job.create(JobConfig.model_validate(cfg))).run()

    asyncio.run(go())


def run_trial(claim: dict[str, Any], grid: dict[str, Any], token: str, hf_user: str, workdir: Path, *,
              harbor: Callable[[dict[str, Any]], None] = harbor_run, environ: MutableMapping[str, str] | None = None) -> Path:
    """Run the claimed cell; returns its trial folder (result.json, agent/hlb-meter.jsonl, trajectory)."""
    rc = {**{k: grid[k] for k in ("reasoning_effort", "provider", "benchmark") if k in grid}, **(claim.get("run_config") or {})}
    run = build_run(claim["model"], rc.get("reasoning_effort", "low"), rc["provider"])  # the attendee's (cli.Session) or the coordinator's
    prepare_env(os.environ if environ is None else environ, token, hf_user, run["run_id"])
    job_name = f"{run['run_id']}-{claim['harness']}-{uuid.uuid4().hex[:6]}"  # its own folder: slots may run one harness at once
    cfg = job_config(run, claim["harness"], claim["task"], claim["cell_id"], Path(workdir), job_name, rc.get("provider"), rc.get("benchmark", "terminal-bench-2.1"))
    harbor(cfg)
    trials = sorted(d for d in (Path(workdir) / job_name).glob("*") if d.is_dir())
    if not trials:
        raise TrialError(f"Harbor left no trial folder in {Path(workdir) / job_name}")
    return trials[0]


def sandbox_id(api: Any, cell_id: str, hf_user: str) -> str | None:
    """The trial's sandbox Job id, found by the cell label in the attendee's namespace; None if the lookup fails."""
    try:
        jobs = list(api.list_jobs(labels={CELL_LABEL: hlb_config.label_value(cell_id)}, namespace=hf_user))
    except Exception:  # noqa: BLE001 - a missing id must not cost the result
        return None
    return jobs[0].id if jobs else None
