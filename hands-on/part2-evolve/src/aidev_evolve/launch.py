"""Laptop side of an evaluation: one HF Job in the attendee's own namespace, records read back from its log.

The Job downloads this version's wheel from Hugging Face (WHEEL_REPO; with the Job's own token), installs it with the `run` extra (Harbor and the Part 1 sandbox
wiring) and runs `aidev-evolve eval-job`; the spec (candidate sources and the trial plan) rides in one
environment variable and the token only in the Job secret `HF_TOKEN` (as in Part 1's `launch`).
"""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aidev_evolve import __version__, evaljob, transport

JOB_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
# THE place the wheel lives: the Hugging Face dataset repository the evaluation Jobs download it from
# (devtools/publish.py uploads it). Not GitHub: hundreds of Jobs share a few egress IPs, and GitHub answered them with
# HTTP 403 (load test, 25 Nov 2026). It is PUBLIC and shared with the Part 1 runner (aidev_lb.launch names the same
# repository): one folder each.
WHEEL_REPO = "agent-harness-workshop/aidev-workshop-wheels"
WHEEL_FOLDER = "evolve"
FLAVOR = "cpu-basic"
JOB_USD_PER_H = 0.01  # cpu-basic
SETUP_S = 10 * 60  # the Job's own install before the trials start, with room
MAX_SPEC_CHARS = 100_000  # a 60 KB variable was probed fine (17 Nov); keep a margin
JOB_LABEL = {"aidev-evolve": "eval"}
RUNNING = {"SCHEDULING", "RUNNING", "STARTING", "PENDING", "QUEUED"}


class EvalJobError(RuntimeError):
    """The evaluation Job ended without sending all its results."""


@dataclass
class EvalResult:
    records: list[dict[str, Any]]
    job_id: str
    job_url: str | None
    seconds: float
    job_usd: float
    log_tail: list[str] = field(default_factory=list)
    missing: int = 0  # trials the spec asked for that sent no record (a crashed Harbor job, a lost log line)


def default_package_url(version: str = __version__) -> str:
    return f"https://huggingface.co/datasets/{WHEEL_REPO}/resolve/main/{WHEEL_FOLDER}/aidev_evolve-{version}-py3-none-any.whl"


def _download(url: str, to: str) -> str:
    """Shell: download with one more try after a short random delay. A Hub URL gets the Job's own token from the
    environment (the secret HF_TOKEN), so the Hub counts the request against the attendee's account, not the shared IP."""
    auth = ' -H "Authorization: Bearer $HF_TOKEN"' if url.startswith("https://huggingface.co/") else ""
    curl = f"curl -fsSL --retry 2{auth} -o {shlex.quote(to)} {shlex.quote(url)}"
    return f"{{ {curl} || {{ sleep $((2 + RANDOM % 7)); {curl}; }}; }}"


def job_script(url: str) -> str:
    """The wheel is the Job's only download besides the PyPI packages and the leaderboard library (git on the Hub):
    the tasks travel in it."""
    wheel = "/tmp/" + url.rsplit("/", 1)[-1]
    return (f'set -e; {_download(url, wheel)}; '
            f'exec uvx --python 3.12 --from "aidev-evolve[run] @ file://{wheel}" aidev-evolve eval-job')


def job_spec(token: str, user: str, spec: dict[str, Any], *, package_url: str | None = None, flavor: str = FLAVOR) -> dict[str, Any]:
    if not spec.get("providers"):
        raise ValueError("the evaluation spec names no providers: pass providers.for_user(user)")
    encoded = evaljob.encode_spec(spec)
    if len(encoded) > MAX_SPEC_CHARS:
        raise ValueError(f"evaluation spec too large for a Job variable ({len(encoded):,} chars > {MAX_SPEC_CHARS:,}): is agent.py huge?")
    return {
        "image": JOB_IMAGE,
        "command": ["bash", "-c", job_script(package_url or default_package_url())],
        "env": {evaljob.SPEC_ENV: encoded,
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",  # litellm's packaged price list, not the one on raw.githubusercontent.com
                "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        "secrets": {"HF_TOKEN": token},
        "flavor": flavor,
        "timeout": int(evaljob.wall_limit_s(float(spec.get("budget_s", 150))) + SETUP_S),
        "labels": dict(JOB_LABEL),
        "namespace": user,
    }


def _stage_running(api: Any, job_id: str, user: str) -> bool:
    try:
        return str(api.inspect_job(job_id=job_id, namespace=user).status.stage).upper().split(".")[-1] in RUNNING
    except Exception:  # noqa: BLE001 - unknown means: try reading the log again
        return True


def run_eval(api: Any, token: str, user: str, spec: dict[str, Any], *, on_record: Callable[[dict[str, Any]], None] = lambda r: None,
             on_job: Callable[[Any], None] = lambda job: None, package_url: str | None = None, flavor: str = FLAVOR, clock: Callable[[], float] = time.monotonic,
             stage_running: Callable[[Any, str, str], bool] = _stage_running, sleep: Callable[[float], None] = time.sleep,
             max_follows: int = 30) -> EvalResult:
    start = clock()
    job = api.run_job(**job_spec(token, user, spec, package_url=package_url, flavor=flavor))
    on_job(job)
    dec = transport.Decoder()
    records: list[dict[str, Any]] = []
    tail: list[str] = []
    expected: int | None = None
    try:
        for _ in range(max_follows):
            for line in api.fetch_job_logs(job_id=job.id, namespace=user, follow=True):
                for rec in dec.feed(line):
                    records.append(rec)
                    on_record(rec)
                if transport.is_done(line):
                    expected = transport.done_count(line)
                    break
                if transport.MARK not in line and line.strip():
                    tail = (tail + [line.rstrip()])[-40:]
            if expected is not None or not stage_running(api, job.id, user):
                break
            sleep(2.0)
    except BaseException:  # Ctrl-C or a crash on the laptop: do not leave the Job (and its sandboxes) running
        try:
            api.cancel_job(job_id=job.id, namespace=user)
        finally:
            raise
    seconds = clock() - start
    if expected is None and not records:
        raise EvalJobError(f"evaluation Job {job.id} ended without its results ({len(records)} received). Last log lines:\n" + "\n".join(tail[-15:]))
    planned = sum(len(r["tasks"]) * int(r.get("trials", 1)) for r in spec["runs"])
    return EvalResult(records, job.id, getattr(job, "url", None), seconds, seconds / 3600 * JOB_USD_PER_H, tail, max(0, planned - len(records)))
