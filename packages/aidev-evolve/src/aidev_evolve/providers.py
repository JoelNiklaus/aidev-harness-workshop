"""Which GLM-5.3-Flash provider serves an attendee, and the fallback inside one evaluation Job.

The list is the packaged `data/providers.json`. Weighted rendezvous hashing: every (user, provider) pair gets a
pseudo-random u in (0, 1) from SHA-256, and providers are ordered by -ln(u) / weight, smallest first. The first is the
attendee's provider, the rest the fallback in order; the order depends on the username only.

Inside a Job (`Pool`, shared by all trials): a call that gets 429/5xx or times out is retried once on the next
provider; after 3 such fallbacks the rest of the Job starts its calls on the next provider. A provider that refuses the
model (HTTP 400 "not supported by provider") is left out for the rest of the Job.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from importlib import resources
from typing import Any

MODEL = "zai-org/GLM-5.3-Flash"
REFERENCE = "zai-org"  # the route name hlb's run record is built with (evaljob.build_run): hlb knows its limits; no model call goes there
FALLBACKS_TO_SWITCH = 3
# Check values for the packaged list (pinned by tests/test_providers.py).
TEST_VECTORS = {"alice": ["particle-ai", "together", "deepinfra", "wafer", "coreweave"], "dave": ["wafer", "particle-ai", "deepinfra", "together", "coreweave"], "JoelNiklaus": ["deepinfra", "together", "particle-ai", "wafer", "coreweave"]}


def load() -> list[dict[str, Any]]:
    """The packaged allowlist."""
    return json.loads(resources.files("aidev_evolve.data").joinpath("providers.json").read_text())["providers"]


def _u(user: str, provider: str) -> float:
    h = hashlib.sha256(f"{user.lower()}\n{provider}".encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2**64


def order(user: str, providers: list[dict[str, Any]]) -> list[str]:
    """The attendee's providers, preferred first. Providers with weight 0 are left out."""
    live = [p for p in providers if p["weight"] > 0]
    return [p["name"] for p in sorted(live, key=lambda p: (-math.log(_u(user, p["name"])) / p["weight"], p["name"]))]


def for_user(user: str) -> list[str]:
    """The attendee's order over the packaged list."""
    return order(user, load())


class Pool:
    """The providers one evaluation Job uses, preferred first; shared by the Job's trial threads."""

    def __init__(self, names: list[str]):
        if not names:
            raise ValueError("a provider pool needs at least one provider")
        self.order = list(names)
        self.refused: set[str] = set()
        self._first, self._since_switch, self.fallbacks_total = 0, 0, 0
        self._lock = threading.Lock()

    def _next(self, after: str) -> str | None:
        """The next provider after `after` in the order that has not refused the model (None: there is no other)."""
        i = self.order.index(after)
        for k in range(1, len(self.order)):
            name = self.order[(i + k) % len(self.order)]
            if name not in self.refused:
                return name
        return None

    def first(self) -> str:
        return self.order[self._first]

    def fallback(self, failed: str) -> str | None:
        """A call on `failed` got 429/5xx or timed out: the provider to retry it on (None when there is no other).
        The third fallback in this Job moves its first provider to the next one."""
        with self._lock:
            nxt = self._next(failed)
            if nxt is None:
                return None
            self.fallbacks_total += 1
            self._since_switch += 1
            if self._since_switch >= FALLBACKS_TO_SWITCH:
                self._first, self._since_switch = self.order.index(self._next(self.first()) or self.first()), 0
            return nxt

    def refuse(self, provider: str) -> str | None:
        """`provider` does not serve the model: leave it out for the rest of the Job. Returns the provider to use
        instead (None when every provider refused)."""
        with self._lock:
            self.refused.add(provider)
            nxt = self._next(provider)
            if self.first() in self.refused and nxt is not None:
                self._first, self._since_switch = self.order.index(self._next(self.first())), 0
            return nxt


_POOLS: dict[tuple[str, ...], Pool] = {}
_POOLS_LOCK = threading.Lock()


def shared_pool(names: list[str]) -> Pool:
    """One pool per provider order in this process: all trials of an evaluation Job share it."""
    with _POOLS_LOCK:
        return _POOLS.setdefault(tuple(names), Pool(names))
