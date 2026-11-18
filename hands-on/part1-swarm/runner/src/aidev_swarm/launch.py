"""`aidev-swarm launch`: one HF Job in the attendee's own namespace runs the swarm trials on HF.

The Job installs this same runner version (with the heavy `run` extra: Harbor and the Harness
Leaderboard wiring) and runs `aidev-swarm run --loop` or `--n K`. Harbor then creates the trial
sandboxes from inside the Job, so the laptop only submits the Job and can close right after.
The attendee's token reaches the Job only as an encrypted Job secret (HF_TOKEN); it is never part
of the command, the environment or any printed line.
"""

from __future__ import annotations

import shlex
from typing import Any

from aidev_swarm import __version__

JOB_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"  # uv + Python 3.12 + git (Harbor fetches tasks and hlb by git) + curl
GITHUB_REPO = "JoelNiklaus/aidev-harness-workshop"  # public: the runner wheel is a release asset, so attendees need no access to anything private
FLAVOR = "cpu-basic"
TRIAL_S = 45 * 60  # a subset trial takes <= ~20 min plus up to 18 min harness install
SETUP_S = 15 * 60  # Job start, package install, slack
LOOP_TIMEOUT_S = 3 * 3600
JOB_LABEL = {"aidev-swarm": "runner"}


def release_tag(version: str = __version__) -> str:
    return f"swarm-v{version}"


def default_package_url(version: str = __version__) -> str:
    return f"https://github.com/{GITHUB_REPO}/releases/download/{release_tag(version)}/aidev_swarm-{version}-py3-none-any.whl"


def timeout_s(n: int | None) -> int:
    return LOOP_TIMEOUT_S if n is None else n * TRIAL_S + SETUP_S


def job_script(url: str, n: int | None) -> str:
    wheel = "/tmp/" + url.rsplit("/", 1)[-1]  # uv needs the wheel's own file name to read its version
    mode = f"--n {n}" if n is not None else "--loop"
    return (f"set -e; curl -fsSL -o {shlex.quote(wheel)} {shlex.quote(url)}; "
            f'exec uvx --python 3.12 --from "aidev-swarm[run] @ file://{wheel}" aidev-swarm run {mode}')


def job_spec(token: str, user: str, coordinator: str, *, n: int | None, package_url: str | None = None) -> dict[str, Any]:
    """Keyword arguments for `HfApi.run_job`: the token appears only under `secrets`."""
    return {
        "image": JOB_IMAGE,
        "command": ["bash", "-c", job_script(package_url or default_package_url(), n)],
        "env": {"AIDEV_SWARM_URL": coordinator, "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        "secrets": {"HF_TOKEN": token},
        "flavor": FLAVOR,
        "timeout": timeout_s(n),
        "labels": dict(JOB_LABEL),
        "namespace": user,
    }


def package_status(url: str) -> int:
    """HTTP status of the public wheel (200: the Job will be able to download it). No credentials are sent."""
    import httpx

    return httpx.head(url, follow_redirects=True, timeout=30).status_code


def submit(token: str, **spec: Any) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token).run_job(**spec)
