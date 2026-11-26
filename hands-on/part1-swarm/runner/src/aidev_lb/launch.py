"""The HF Job `aidev-lb run` starts for one harness, in the attendee's own namespace.

The Job installs this same runner version with the heavy `run` extra (Harbor and the Harness Leaderboard wiring) and
runs `aidev-lb job`: every trial of the harness at once, each in its own HF Sandbox. The attendee's token reaches the
Job only as an encrypted Job secret (HF_TOKEN); it is never part of the command, the environment or a printed line.
"""

from __future__ import annotations

import base64
import shlex
from pathlib import Path
from typing import Any

from aidev_lb import __version__

JOB_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"  # uv + Python 3.12 + git (uv fetches hlb by git from the Hub) + curl
# THE place the runner wheel lives: the Hugging Face dataset repository the Jobs download it from (devtools/publish.py
# uploads it; `run` checks it before it starts a Job). Not
# GitHub: hundreds of Jobs share a few egress IPs, and GitHub answered them with HTTP 403 (load test, 25 Nov 2026).
# It is PUBLIC and shared with the Part 2 package (aidev_evolve.launch names the same repository): one folder each.
WHEEL_REPO = "agent-harness-workshop/aidev-workshop-wheels"
WHEEL_FOLDER = "lb"
FLAVOR = "cpu-basic"  # the Job only orchestrates: about 70 MB and little CPU per trial
# Above the longest trial (15 min agent timeout, up to 18 min harness install, verifier), so Harbor always ends its own
# sandboxes before the Job is cut off. `run` cancels the Job much earlier, when its wait is over.
TIMEOUT_S = 45 * 60
JOB_LABEL = {"aidev-lb": "runner"}
RUN_LABEL = "aidev-lb-run"  # on the Jobs and the trial sandboxes of one `run` command
WHEEL_SECRET = "AIDEV_LB_WHEEL_B64"


def default_package_url(version: str = __version__) -> str:
    return f"https://huggingface.co/datasets/{WHEEL_REPO}/resolve/main/{WHEEL_FOLDER}/aidev_lb-{version}-py3-none-any.whl"


def is_url(package: str) -> bool:
    return package.startswith(("http://", "https://"))


def _download(url: str, to: str) -> str:
    """Shell: download with one more try after a short random delay. A Hub URL gets the Job's own token from the
    environment (the secret HF_TOKEN), so the Hub counts the request against the attendee's account, not the shared IP."""
    auth = ' -H "Authorization: Bearer $HF_TOKEN"' if url.startswith("https://huggingface.co/") else ""
    curl = f"curl -fsSL --retry 2{auth} -o {shlex.quote(to)} {shlex.quote(url)}"
    return f"{{ {curl} || {{ sleep $((2 + RANDOM % 7)); {curl}; }}; }}"


def job_script(package: str, harness: str, tasks: list[str], user: str) -> str:
    """`package`: the wheel's URL, or a local wheel file (an unreleased version: its bytes travel as a Job secret).
    The wheel is the Job's only download besides the PyPI packages and the leaderboard library (git on the Hub):
    the tasks travel in it."""
    wheel = "/tmp/" + package.rsplit("/", 1)[-1]  # uv needs the wheel's own file name to read its version
    fetch = _download(package, wheel) if is_url(package) else f'printf %s "${WHEEL_SECRET}" | base64 -d > {shlex.quote(wheel)}'
    return (f"set -e; {fetch}; "
            f'exec uvx --python 3.12 --from "aidev-lb[run] @ file://{wheel}" '
            f"aidev-lb job --harness {shlex.quote(harness)} --tasks {shlex.quote(','.join(tasks))} --user {shlex.quote(user)}")


def job_spec(token: str, user: str, harness: str, tasks: list[str], run_key: str, *, package: str | None = None,
             bucket: str | None = None) -> dict[str, Any]:
    """Keyword arguments for `HfApi.run_job`: the token appears only under `secrets`. `bucket`: the attendee's results
    bucket, when the run has one."""
    package = package or default_package_url()
    secrets = {"HF_TOKEN": token}
    if not is_url(package):
        secrets[WHEEL_SECRET] = base64.b64encode(Path(package).read_bytes()).decode()
    return {
        "image": JOB_IMAGE,
        "command": ["bash", "-c", job_script(package, harness, tasks, user)],
        "env": {"AIDEV_LB_RUN": run_key, **({"AIDEV_LB_BUCKET": bucket} if bucket else {}),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",  # litellm's packaged price list, not the one on raw.githubusercontent.com
                "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        "secrets": secrets,
        "flavor": FLAVOR,
        "timeout": TIMEOUT_S,
        "labels": {**JOB_LABEL, RUN_LABEL: run_key},
        "namespace": user,
    }


def package_status(url: str, token: str) -> int:
    """HTTP status of the wheel (200: the Job will be able to download it). As in the Job's download, the token is
    sent to huggingface.co only."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if url.startswith("https://huggingface.co/") else {}
    return httpx.head(url, headers=headers, follow_redirects=True, timeout=30).status_code


def submit(token: str, **spec: Any) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token).run_job(**spec)


def job_logs(token: str, user: str, job_id: str):
    """The Job's log lines from its start, following until the Job ends or the stream breaks."""
    from huggingface_hub import HfApi

    return HfApi(token=token).fetch_job_logs(job_id=job_id, namespace=user, follow=True)


def cancel_sandboxes(token: str, user: str, run_key: str) -> None:
    """Stop what a `run` command left running: its trial sandboxes are Jobs of their own, labelled with the run key
    (as are the runner Jobs, which `run` cancels by id first)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for job in api.list_jobs(status="RUNNING", labels={RUN_LABEL: run_key}, namespace=user):
        api.cancel_job(job_id=job.id, namespace=user)
