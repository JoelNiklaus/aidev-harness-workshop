"""`aidev-swarm check`: find out at home whether this HF account can do what the workshop needs.

It makes one call to the workshop model through HF Inference Providers, runs one tiny cpu-basic HF Job
(what `launch` starts) and one command in an HF Sandbox (where trials run), then reports every failure
with its fix. Together a few seconds of cpu-basic, well under $0.01. It claims no cell. The Job gets no
token: it only prints a line.
"""

from __future__ import annotations

from typing import Any

ROUTER = "https://router.huggingface.co/v1/chat/completions"
IMAGE = "python:3.12-slim"
LABEL = {"aidev-swarm": "check"}
DONE = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def job_spec(user: str) -> dict[str, Any]:
    """Keyword arguments for `HfApi.run_job`: no secrets, 5-minute cap."""
    return {"image": IMAGE, "command": ["python", "-c", "print('aidev-swarm check: this HF Job ran')"],
            "flavor": "cpu-basic", "timeout": 300, "labels": dict(LABEL), "namespace": user}


class ProviderBusy(RuntimeError):
    """The provider answered 429 or 5xx: worth trying the next provider of the attendee's list."""


def chat(token: str, model: str, provider: str) -> str:
    """One short reply from `model` on `provider`, billed to the token's account. Raises ProviderBusy on 429/5xx."""
    import httpx

    r = httpx.post(ROUTER, headers={"Authorization": f"Bearer {token}"}, timeout=120, json={
        "model": f"{model}:{provider}", "max_tokens": 200, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
    if r.status_code == 429 or r.status_code >= 500:
        raise ProviderBusy(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def sandbox_echo(token: str, user: str) -> str:
    """Start one HF Sandbox the way Harbor does for a trial, run one command in it, and stop it."""
    from huggingface_hub import Sandbox

    with Sandbox.create(image=IMAGE, namespace=user, token=token, idle_timeout=120, labels=dict(LABEL)) as sbx:
        return sbx.run("echo sandbox ok", timeout=60).stdout.strip()


def job_stage(token: str, user: str, job_id: str) -> str:
    from huggingface_hub import HfApi

    stage = HfApi(token=token).inspect_job(job_id=job_id, namespace=user).status.stage
    return getattr(stage, "value", stage)  # JobStage is a str Enum; str() would give "JobStage.COMPLETED"


def cancel_job(token: str, user: str, job_id: str) -> None:
    from huggingface_hub import HfApi

    HfApi(token=token).cancel_job(job_id=job_id, namespace=user)
