"""`aidev-lb check`: find out at home whether this HF account can do what the workshop needs.

It makes one call to the workshop model through HF Inference Providers, runs one tiny cpu-basic HF Job
(what `run` starts) and one command in an HF Sandbox (where trials run), then reports every failure
with its fix. Together a few seconds of cpu-basic, well under $0.01. The Job gets no token: it only prints a line.
"""

from __future__ import annotations

from typing import Any

from aidev_lb.providers import is_provider_error

ROUTER = "https://router.huggingface.co/v1/chat/completions"
IMAGE = "python:3.12-slim"
LABEL = {"aidev-lb": "check"}
DONE = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
CHAT_TIMEOUT_S = 60  # a provider that takes longer for one word is no use to a trial; the next one is tried


def job_spec(user: str) -> dict[str, Any]:
    """Keyword arguments for `HfApi.run_job`: no secrets, 5-minute cap."""
    return {"image": IMAGE, "command": ["python", "-c", "print('aidev-lb check: this HF Job ran')"],
            "flavor": "cpu-basic", "timeout": 300, "labels": dict(LABEL), "namespace": user}


class ProviderBusy(RuntimeError):
    """The provider did not serve the call (providers.is_provider_error: rate limit, outage or refusal): worth trying
    the next provider of the attendee's list."""


def chat(token: str, model: str, provider: str) -> str:
    """One short reply from `model` on `provider`, billed to the token's account. Raises ProviderBusy on a provider error."""
    import httpx

    try:
        r = httpx.post(ROUTER, headers={"Authorization": f"Bearer {token}"}, timeout=CHAT_TIMEOUT_S, json={
            "model": f"{model}:{provider}", "max_tokens": 200, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
    except httpx.TransportError as e:  # no answer in time, or no connection
        raise ProviderBusy(f"{type(e).__name__}: {str(e)[:200]}") from e
    if is_provider_error(r.status_code, r.text):
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
