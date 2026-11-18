"""The model call: one route, fixed by the experiment (GLM-5.3-Flash on the HF router, effort low).

Runs in the HF Job with the attendee's token. Rate limits (429), server errors (5xx) and network errors
are retried with backoff while the trial has time left. A 4xx the agent caused (a malformed history) is an
`LLMError` the agent sees; an account problem (401/402/403: token, credits) is `InfraError`, which ends the
trial as infra (not scored) and which an agent cannot catch with `except Exception`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from aidev_evolve.runtime import InfraError

ROUTER = "https://router.huggingface.co/v1/chat/completions"
ROUTE = "zai-org/GLM-5.3-Flash:zai-org"
EFFORT = "low"
KEEP = ("role", "content", "tool_calls")


class LLMError(RuntimeError):
    """The router refused the request (a harness bug, e.g. a malformed message history)."""


class RouterLLM:
    def __init__(self, token: str, *, route: str = ROUTE, effort: str = EFFORT, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic, url: str = ROUTER):
        self._token, self._route, self._effort, self._url = token, route, effort, url
        self._client = client or httpx.Client()
        self._sleep, self._clock = sleep, clock

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, timeout: float, **params: Any) -> tuple[dict[str, Any], dict[str, int]]:
        body: dict[str, Any] = {"model": self._route, "messages": messages, "reasoning_effort": self._effort, **params}
        if tools:
            body["tools"] = tools
        start, waited, backoff = self._clock(), 0.0, 2.0
        while True:
            left = timeout - (self._clock() - start) - waited
            if left <= 0:
                raise TimeoutError("no time left for the model call")
            try:
                r = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self._token}"}, timeout=min(left, 600.0))
            except httpx.TimeoutException:
                raise TimeoutError("model call timed out") from None
            except httpx.TransportError:
                r = None
            if r is not None and r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                u = data.get("usage") or {}
                usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                         "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}
                return {k: msg.get(k) for k in KEEP if k in msg}, usage
            if r is not None and r.status_code in (401, 402, 403):
                raise InfraError(f"router HTTP {r.status_code}: {r.text[:300]}")
            if r is not None and r.status_code != 429 and r.status_code < 500:
                raise LLMError(f"router HTTP {r.status_code}: {r.text[:500]}")
            if waited + (self._clock() - start) + backoff >= timeout:
                raise TimeoutError(f"router busy (last HTTP {r.status_code if r is not None else 'network error'})")
            self._sleep(backoff)
            waited += backoff
            backoff = min(backoff * 2, 20.0)
