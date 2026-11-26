"""Run one cell (harness, task) as one Harbor trial on HF Sandbox, under the attendee's own HF token.

Reuses the Harness Leaderboard's wiring (`hlb`, pinned to a Space revision): the per-harness Harbor agent configs,
the HF Sandbox environment with the in-sandbox metering proxy, and the key custody (harnesses get a per-run dummy key;
only the meter holds the real token).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from hlb import config as hlb_config
from hlb.worker import hand_out_sandbox_key, harbor_job_config

from aidev_lb import nogithub
from aidev_lb.launch import RUN_LABEL

CELL_LABEL = "aidev-lb-cell"
# One run per process, as a leaderboard worker: hlb hands the harnesses their dummy key through os.environ, which the
# parallel trials (threads) of one Job share, so every trial of the process must use the same run id and key.
RUN_ID = f"lb-{uuid.uuid4().hex[:8]}"
# What a trial reports while it runs, in order (Harbor's hooks: environment started, agent started, verification started).
PHASES = ("sandbox_started", "harness_installed", "agent_finished")


# Longest a trial may take before it is stopped as an infrastructure error: above the tasks' 15-minute agent limit plus
# a slow harness install and the verifier, below the Job's timeout. One trial in 250 hung for 17 minutes in the load
# test of 25 Nov 2026. AIDEV_LB_TRIAL_WALL_S overrides it.
TRIAL_WALL_S = 25 * 60


class TrialError(RuntimeError):
    """Harbor ended without leaving a trial folder."""


class TrialWallLimit(RuntimeError):
    """The trial did not end within its wall limit and was stopped."""


def wall_limit_s(environ: Mapping[str, str]) -> float:
    return float(environ.get("AIDEV_LB_TRIAL_WALL_S") or TRIAL_WALL_S)


async def within(run: Awaitable[Any], limit_s: float) -> Any:
    """Await `run`, cancelling it at the limit (Harbor stops the trial's sandbox when it is cancelled)."""
    try:
        return await asyncio.wait_for(run, limit_s)
    except TimeoutError:
        raise TrialWallLimit(f"no end after {limit_s:.0f} s, the wall limit of a trial: stopped") from None


def build_run(model: str, effort: str, providers: list[str], meter: dict[str, dict[str, Any]] | None = None, run_key: str = "") -> dict[str, Any]:
    """The run record Harbor's job config is built from, as the leaderboard's launcher records one. Harness configs
    name the model's reference route (the leaderboard's, with its limits); the pool is the attendee's `providers` in
    their order, any router provider of the model: each trial's meter rewrites the route to its provider and sends a
    rate-limited call on to the next one. `meter`: per-provider meter settings (default: max_completion_tokens goes
    as max_tokens, which every route enforces). `run_key`: the laptop command's key, a label on every sandbox."""
    ref = hlb_config.models()[model]["default_provider"]
    settings = {"max_completion_tokens_as_max_tokens": True}
    return {"run_id": RUN_ID, "run_key": run_key, "model": model, "provider": ref, "reasoning_effort": effort,
            "benchmark_setups": {b: hlb_config.benchmark_setup(b) for b in hlb_config.benchmarks()},
            "route": hlb_config.route(model, ref), "model_limits": hlb_config.model_limits(model, ref),
            "meter_settings": hlb_config.meter_settings(model, ref), "npm_before": {},
            "providers": {p: {"route_model": f"{model}:{p}", "meter_settings": {**settings, **(meter or {}).get(p, {})}} for p in providers}}


def prepare_env(environ: MutableMapping[str, str], token: str, hf_user: str, run_id: str) -> None:
    """The worker environment hlb expects: the token for sandboxes (HF_TOKEN) and for the meter
    (HF_INFERENCE_TOKEN), the harnesses' dummy key, and the attendee's namespace, where hlb looks up
    and cancels its sandbox Jobs (hlb hardcodes the leaderboard owner there)."""
    environ["HF_TOKEN"] = token
    environ["HF_INFERENCE_TOKEN"] = token
    hand_out_sandbox_key(run_id, environ)
    hlb_config.OWNER = hf_user


def job_config(run: dict[str, Any], harness: str, task: str, cell_id: str, jobs_dir: Path, job_name: str, provider: str,
               benchmark: str = "terminal-bench-2.1") -> dict[str, Any]:
    """The leaderboard's Harbor job config for one trial, with two changes that keep GitHub off its path (nogithub):
    the task is read from the folder unpacked from the wheel, not cloned, and the sandbox environment installs
    harnesses without nvm's installer."""
    cfg = harbor_job_config(run, benchmark, [harness], [task], 1, job_name, Path(jobs_dir),
                            {CELL_LABEL: hlb_config.label_value(cell_id), RUN_LABEL: hlb_config.label_value(run["run_key"])},
                            concurrency=1, provider=provider)
    cfg["datasets"] = [{"path": str(nogithub.tasks_dir()), "task_names": [task]}]
    cfg["environment"]["import_path"] = "aidev_lb.nogithub:WorkshopSandbox"
    return cfg


def harbor_run(cfg: dict[str, Any], on_event: Callable[[str], None]) -> None:
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    def report(phase: str):
        async def hook(event: Any) -> None:
            on_event(phase)

        return hook

    async def go() -> None:
        job = await Job.create(JobConfig.model_validate(cfg))
        job.on_environment_started(report(PHASES[0])).on_agent_started(report(PHASES[1])).on_verification_started(report(PHASES[2]))
        await within(job.run(), wall_limit_s(os.environ))

    asyncio.run(go())


def run_trial(cell: dict[str, Any], run: dict[str, Any], provider: str, token: str, hf_user: str, workdir: Path, *,
              on_event: Callable[[str], None] = lambda phase: None,
              harbor: Callable[[dict[str, Any], Callable[[str], None]], None] = harbor_run,
              environ: MutableMapping[str, str] | None = None, benchmark: str = "terminal-bench-2.1") -> Path:
    """Run the cell on `provider` (one of the run's pool); returns its trial folder (result.json,
    agent/hlb-meter.jsonl, trajectory). `on_event` gets each of PHASES as the trial reaches it."""
    prepare_env(os.environ if environ is None else environ, token, hf_user, run["run_id"])
    job_name = f"{run['run_id']}-{cell['harness']}-{uuid.uuid4().hex[:6]}"  # its own folder: trials of one harness run at once
    harbor(job_config(run, cell["harness"], cell["task"], cell["cell_id"], Path(workdir), job_name, provider, benchmark), on_event)
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
