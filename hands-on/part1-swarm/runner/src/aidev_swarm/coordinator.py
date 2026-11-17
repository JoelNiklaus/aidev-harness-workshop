"""HTTP client for the swarm coordinator (PROTOCOL.md: claim, heartbeat, submit, release).

The HF token goes only into the Authorization header; it never appears in an error message or log line.
Transient failures (connection errors, 5xx, 429) are retried with capped exponential backoff until a
per-call time budget runs out. The HF edge in front of the Space rate-limits per IP and can block an IP
for minutes (PROTOCOL.md, Rules), so a finished trial's submit keeps trying for 35 minutes; a heartbeat
gives up after a minute, because the next beat (every 2 min) tries again and the lease lasts 40 minutes.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable
from typing import Any

import httpx


class CoordinatorError(Exception):
    """The coordinator refused or could not be reached within the call's budget."""


class LeaseLost(CoordinatorError):
    """409: the cell's lease or result belongs to someone else now (it expired and was handed out again)."""


class AtLimit(CoordinatorError):
    """409 on claim: this user already holds as many leases as allowed (`held` lists them)."""

    def __init__(self, msg: str, held: list[str]):
        super().__init__(msg)
        self.held = held


RETRY_STATUS = {429, 500, 502, 503, 504}
# Seconds each call keeps retrying transient failures. Submit: >= 30 min (PROTOCOL.md rule).
BUDGET_S = {"claim": 10 * 60, "heartbeat": 60, "submit": 35 * 60, "release": 2 * 60}


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return float(r.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _json(r: httpx.Response) -> dict[str, Any]:
    try:
        body = r.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


class Coordinator:
    def __init__(self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None, retries: int | None = None,
                 backoff_s: float = 2.0, max_backoff_s: float = 60.0, timeout_s: float = 30.0, budgets: dict[str, float] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        """`retries` caps the attempts (tests use 1 to fail fast); by default only the time budget does."""
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {token}"}, transport=transport, timeout=timeout_s)
        self.retries, self.backoff_s, self.max_backoff_s = retries, backoff_s, max_backoff_s
        self.budgets = {**BUDGET_S, **(budgets or {})}
        self._sleep, self._clock = sleep, clock

    def _post(self, verb: str, path: str, *, params: dict[str, Any] | None = None, json: Any = None) -> httpx.Response:
        deadline = self._clock() + self.budgets[verb]
        last, attempt = "", 0
        while True:
            attempt += 1
            wait = min(self.max_backoff_s, self.backoff_s * (2 ** (attempt - 1)))
            try:
                r = self._http.post(path, params=params, json=json)
            except httpx.TransportError as e:
                last = type(e).__name__
            else:
                if r.status_code == 409:
                    body = _json(r)
                    if verb == "claim" and "held" in body:
                        raise AtLimit(f"{path}: {str(body.get('detail', ''))[:200]}", list(body["held"]))
                    raise LeaseLost(f"{path}: {r.text[:200]}")
                if r.status_code not in RETRY_STATUS:
                    if r.status_code >= 400:
                        raise CoordinatorError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
                    return r
                last = f"HTTP {r.status_code}"
                wait = _retry_after(r) or wait
            if (self.retries is not None and attempt >= self.retries) or self._clock() + wait > deadline:
                raise CoordinatorError(f"{path}: gave up after {attempt} attempts ({last})")
            self._sleep(wait)

    @staticmethod
    def _cell(cell_id: str) -> str:
        return urllib.parse.quote(cell_id, safe="")

    def claim(self, n: int = 1) -> dict[str, Any] | list[dict[str, Any]] | None:
        r = self._post("claim", "/api/claim", params={"n": n} if n != 1 else None)
        return None if r.status_code == 204 else r.json()

    def heartbeat(self, cell_id: str) -> None:
        self._post("heartbeat", f"/api/heartbeat/{self._cell(cell_id)}")

    def submit(self, cell_id: str, result: dict[str, Any]) -> None:
        self._post("submit", f"/api/submit/{self._cell(cell_id)}", json=result)

    def release(self, cell_id: str) -> None:
        self._post("release", f"/api/release/{self._cell(cell_id)}")
