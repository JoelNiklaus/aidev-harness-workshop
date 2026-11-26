"""`aidev-lb job`: what one HF Job runs. Every trial of one harness at once, each in its own HF Sandbox.

It first finds a provider that answers (one short model call per provider of the attendee's order, so a provider
that refuses the model costs seconds, not a trial), then starts all trials in threads. A trial whose every model call
ended in a provider error runs again on the next provider. A finished trial goes to the results store; a trial that
says nothing about the harness (infrastructure error) is reported as such and never stored as a fail.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aidev_lb import __version__, catalog, check, events, providers, store


def _run_trial(cell, run, provider, token, user, workdir, *, on_event):
    from aidev_lb.trial import run_trial

    return run_trial(cell, run, provider, token, user, workdir, on_event=on_event)


def _sandbox_id(token: str) -> Callable[[str, str], str | None]:
    def lookup(cell_id: str, user: str) -> str | None:
        from huggingface_hub import HfApi

        from aidev_lb.trial import sandbox_id

        return sandbox_id(HfApi(token=token), cell_id, user)

    return lookup


def _cancel_sandbox(token: str) -> Callable[[str, str, str], None]:
    def cancel(cell_id: str, user: str, run_key: str) -> None:
        from hlb.config import label_value
        from huggingface_hub import HfApi

        from aidev_lb.launch import RUN_LABEL
        from aidev_lb.trial import CELL_LABEL

        api = HfApi(token=token)
        for sandbox in api.list_jobs(status="RUNNING", labels={CELL_LABEL: label_value(cell_id), RUN_LABEL: label_value(run_key)}, namespace=user):
            api.cancel_job(job_id=sandbox.id, namespace=user)

    return cancel


def _exit(code: int) -> None:
    sys.stdout.flush()
    os._exit(code)


@dataclass
class Deps:
    chat: Callable[[str, str, str], str] = check.chat
    run_trial: Callable[..., Path] = _run_trial
    sandbox_id: Callable[[str, str], str | None] | None = None
    upload: store.Upload = store._upload
    # A trial stops itself at its wall limit (trial.within). One that is stuck where no cancellation reaches it is
    # given up `grace_s` later: an infrastructure error, its sandbox cancelled, and the process ends itself (`exit`),
    # because a stuck thread cannot be stopped and would keep the Job alive until its timeout.
    wall_s: float | None = None  # default: trial.wall_limit_s
    grace_s: float = 300.0
    cancel_sandbox: Callable[[str, str, str], None] | None = None
    exit: Callable[[int], None] = _exit


def answering(order: list[str], chat: Callable[[str], str]) -> tuple[list[str], list[dict[str, str]]]:
    """The providers from the first one that answers a short call on, and the ones skipped before it with the reason.
    A failure that is not the provider's (no credits, a token without the permission) ends the search: no provider
    would answer."""
    skipped = []
    for i, provider in enumerate(order):
        try:
            chat(provider)
        except check.ProviderBusy as e:
            skipped.append({"provider": provider, "why": str(e)[:120]})
        except Exception as e:  # noqa: BLE001
            return [], [*skipped, {"provider": provider, "why": f"{type(e).__name__}: {str(e)[:120]}"}]
        else:
            return order[i:], skipped
    return [], skipped


def run(harness: str, tasks: list[str], token: str, user: str, workdir: Path, run_key: str, deps: Deps | None = None,
        out: Callable[[str], None] = print, bucket: str | None = None) -> int:
    """`bucket`: the attendee's results bucket, when the run has one; each result is also written there."""
    from aidev_lb.result import build_result, infra_error, meter_text
    from aidev_lb.trial import build_run, wall_limit_s

    deps = deps or Deps()
    spec = catalog.spec()
    if harness not in {h["name"] for h in catalog.harnesses()}:
        raise ValueError(f"unknown harness {harness!r}")
    if unknown := [t for t in tasks if t not in spec["tasks"]]:
        raise ValueError(f"unknown tasks {unknown}")
    model, lock = spec["model"], threading.Lock()

    given_up: set[str] = set()  # cells whose trial hung past the wall limit: already reported, nothing more is said

    def emit(event: str, **fields: Any) -> None:
        with lock:  # one whole line at a time
            if fields.get("cell_id") not in given_up:
                out(events.line(event, **fields))

    cells = [{"cell_id": catalog.cell_id(harness, t), "harness": harness, "task": t, "trial": 0, "model": model} for t in tasks]
    names = lambda cell: {k: cell[k] for k in ("cell_id", "harness", "task")}  # noqa: E731
    emit("job_started", harness=harness, tasks=tasks, runner_version=__version__)
    listed = providers.load()
    pool, skipped = answering(providers.order(user, listed), lambda p: deps.chat(token, model, p))
    if not pool:
        for cell in cells:
            emit("infra", **names(cell), error=f"no provider answered the model call ({'; '.join(s['provider'] + ': ' + s['why'] for s in skipped)})")
        emit("job_done", harness=harness)
        return 1
    emit("provider", harness=harness, provider=pool[0], skipped=skipped)
    meter = {p["name"]: {"max_completion_tokens_as_max_tokens": p["max_completion_tokens_as_max_tokens"]} for p in listed
             if "max_completion_tokens_as_max_tokens" in p}
    run_record = build_run(model, spec["reasoning_effort"], pool, meter, run_key)
    lookup = deps.sandbox_id or _sandbox_id(token)
    bucket_failed = threading.Event()

    def keep(result: dict[str, Any]) -> None:
        with lock:
            if result["cell_id"] in given_up:
                return
            try:
                store.write_result(result, out=out, bucket=None if bucket_failed.is_set() else bucket, token=token, upload=deps.upload)
            except Exception as e:  # noqa: BLE001 - the line is out; the bucket copy is extra
                bucket_failed.set()  # say it once and stop trying: every later write would fail the same way
                out(events.line("store_error", harness=harness, bucket=bucket, error=f"{type(e).__name__}: {str(e)[:200]}"))

    def one(cell: dict[str, Any]) -> None:
        for i, provider in enumerate(pool):
            try:
                trial_dir = deps.run_trial(cell, run_record, provider, token, user, workdir, on_event=lambda phase: emit(phase, **names(cell)))
                result = build_result(cell, trial_dir, catalog.prices()[model], hf_user=user, runner_version=__version__, provider=provider,
                                      sandbox_id=lookup(cell["cell_id"], user))
                why = infra_error(result, meter_text(trial_dir))
            except Exception as e:  # noqa: BLE001 - one trial's crash must not take the others down
                emit("infra", **names(cell), error=f"trial crashed ({type(e).__name__}: {str(e)[:200]})")
                return
            if why is None:
                keep(result)
                return
            if result["n_provider_errors"] and i + 1 < len(pool):
                emit("retry", **names(cell), provider=pool[i + 1], why=why)
                continue
            emit("infra", **names(cell), error=why, provider=provider)
            return

    wall = wall_limit_s(os.environ) if deps.wall_s is None else deps.wall_s
    threads = ThreadPoolExecutor(max_workers=len(cells), thread_name_prefix="trial")
    trials = {threads.submit(one, cell): cell for cell in cells}
    done, stuck = wait(trials, timeout=wall + deps.grace_s)
    for trial in done:
        trial.result()
    for cell in (trials[t] for t in stuck):
        emit("infra", **names(cell), error=f"no end after {wall + deps.grace_s:.0f} s: the trial hung and was given up")
        with lock:
            given_up.add(cell["cell_id"])
        try:
            (deps.cancel_sandbox or _cancel_sandbox(token))(cell["cell_id"], user, run_key)
        except Exception:  # noqa: BLE001 - the Job's end stops what is left
            pass
    threads.shutdown(wait=False)
    emit("job_done", harness=harness)
    if stuck:
        deps.exit(0)
    return 0
