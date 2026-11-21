"""Client for the room's /evolve board (API in PROTOCOL.md; the coordinator owner builds the server).

The board is a display: nothing in the loop waits for it. `QueuedBoard` writes every event to an outbox file
in the run folder first and then tries to deliver the whole outbox in order, so events survive a Wi-Fi drop,
a sleeping laptop and a restart (`run --resume`). The server is idempotent on (user, run_id, kind, iteration),
so a retried event replaces, never duplicates. Every event names the client version in its User-Agent: the board takes
events only from the frozen workshop version or newer (HTTP 426 otherwise).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from aidev_evolve import __version__

SCHEMA = "aidev-evolve/1"
ROOM_URL = "https://agent-harness-workshop-aidev-swarm.hf.space"  # the Part 1 coordinator Space (runner cli.DEFAULT_COORDINATOR)
RETRYABLE_4XX = (401, 408, 429)  # a token check that timed out, a slow request, a rate limit: try again later


def default_url() -> str:
    """The board to post to: $AIDEV_BOARD (shared with the Part 1 runner; a staging Space for a dry run), else ROOM_URL."""
    return os.environ.get("AIDEV_BOARD") or ROOM_URL


def page_url(url: str = ROOM_URL) -> str:
    return url.rstrip("/") + "/evolve"


def older(version: str, minimum: str) -> bool:
    """`version` < `minimum`, compared as numbers (0.5.10 is newer than 0.5.2)."""
    def parse(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))
    return parse(version) < parse(minimum)


def version_problem(url: str, *, http: httpx.Client | None = None, timeout_s: float = 5.0) -> str | None:
    """One line when the board accepts only a newer aidev-evolve than this one (it answers this version's events with
    HTTP 426); None when all is well, and also when the board cannot be asked: it is a display, never a blocker."""
    try:
        r = (http or httpx.Client()).get(f"{url.rstrip('/')}/api/versions", timeout=timeout_s)
        need = r.json().get("aidev-evolve") if r.status_code == 200 else None
        if isinstance(need, str) and older(__version__, need):
            return (f"this aidev-evolve ({__version__}) is older than the workshop board accepts ({need}): your run would not "
                    f"appear on the board. Copy the command again from {url.rstrip('/')}/evolve/join")
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return None


class Board:
    def __init__(self, url: str, token: str, *, client: httpx.Client | None = None, timeout_s: float = 10.0):
        self._url, self._token, self._timeout = url.rstrip("/"), token, timeout_s
        self._client = client or httpx.Client()

    def submit(self, event: dict[str, Any]) -> None:
        r = self._client.post(f"{self._url}/api/evolve/submit", json={"schema": SCHEMA, **event},
                              headers={"Authorization": f"Bearer {self._token}", "User-Agent": f"aidev-evolve/{__version__}"}, timeout=self._timeout)
        r.raise_for_status()


class QueuedBoard:
    """Outbox in front of a Board. `submit` and `flush` return how many events still wait for delivery."""

    def __init__(self, inner: Board, outbox: Path):
        self.inner, self.outbox = inner, Path(outbox)
        self.last_error: str | None = None

    def _pending(self) -> list[dict[str, Any]]:
        if not self.outbox.exists():
            return []
        return [json.loads(line) for line in self.outbox.read_text().splitlines() if line.strip()]

    def _store(self, events: list[dict[str, Any]]) -> None:
        self.outbox.parent.mkdir(parents=True, exist_ok=True)
        self.outbox.write_text("".join(json.dumps(e) + "\n" for e in events))

    def _deliver(self) -> int:
        pending = self._pending()
        while pending:
            try:
                self.inner.submit(pending[0])
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if 400 <= code < 500 and code not in RETRYABLE_4XX:
                    self.last_error = f"board refused an update (HTTP {code})"
                    pending.pop(0)  # it will never succeed: do not block the queue behind it
                    continue
                self.last_error = f"HTTP {code}"
                break
            except httpx.HTTPError as e:
                self.last_error = type(e).__name__
                break
            pending.pop(0)
        self._store(pending)
        return len(pending)

    def submit(self, event: dict[str, Any]) -> int:
        self._store(self._pending() + [event])
        return self._deliver()

    def flush(self, retries: int = 1, sleep: Callable[[float], None] = time.sleep, backoff_s: float = 5.0) -> int:
        """Deliver what waits; with retries, wait backoff_s, 2 x backoff_s, ... between attempts."""
        left = self._deliver()
        for k in range(1, retries):
            if not left:
                break
            sleep(backoff_s * k)
            left = self._deliver()
        return left
