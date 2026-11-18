"""Client for the room's /evolve board (API proposed in PROTOCOL.md; the coordinator owner builds the server).

Best effort: the loop catches every error, so a missing or slow board never stops an attendee's run.
"""

from __future__ import annotations

from typing import Any

import httpx

SCHEMA = "aidev-evolve/1"
ROOM_URL = "https://agent-harness-workshop-aidev-swarm.hf.space"  # the Part 1 coordinator Space (runner cli.DEFAULT_COORDINATOR)


class Board:
    def __init__(self, url: str, token: str, *, client: httpx.Client | None = None, timeout_s: float = 10.0):
        self._url, self._token, self._timeout = url.rstrip("/"), token, timeout_s
        self._client = client or httpx.Client()

    def submit(self, event: dict[str, Any]) -> None:
        r = self._client.post(f"{self._url}/api/evolve/submit", json={"schema": SCHEMA, **event},
                              headers={"Authorization": f"Bearer {self._token}"}, timeout=self._timeout)
        r.raise_for_status()
