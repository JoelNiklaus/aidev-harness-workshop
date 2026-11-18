"""Which GLM-5.3-Flash provider serves an attendee, and the fallback inside one evaluation Job.

The list is the workshop's one allowlist (`data/providers.json`, a copy of `../part1-swarm/config/providers.json`;
a test compares them). Weighted rendezvous hashing, as in the Part 1 runner (`aidev_swarm/providers.py`): every
(user, provider) pair gets a pseudo-random u in (0, 1) from SHA-256, and providers are ordered by -ln(u) / weight,
smallest first. The first is the attendee's provider, the rest the fallback in order. Because the order depends on
the username only, it is independent of the attendee's strategy and cannot bias the strategy ranking.

Inside a Job (`Pool`, shared by all trials): a call that gets 429/5xx or times out is retried once on the next
provider; after 3 such fallbacks the rest of the Job starts its calls on the next provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from importlib import resources
from typing import Any

MODEL = "zai-org/GLM-5.3-Flash"
REFERENCE = "zai-org"  # the leaderboard's reference provider: used when a spec names none (devtools sweeps)
FALLBACKS_TO_SWITCH = 3
# Cross-implementation check values for the 3:1 novita/zai-org list (../part1-swarm/PROTOCOL.md "Providers").
TEST_VECTORS = {"alice": ["novita", "zai-org"], "dave": ["zai-org", "novita"], "JoelNiklaus": ["novita", "zai-org"]}


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
        self._first, self._since_switch, self.fallbacks_total = 0, 0, 0
        self._lock = threading.Lock()

    def first(self) -> str:
        return self.order[self._first]

    def fallback(self, failed: str) -> str | None:
        """A call on `failed` got 429/5xx or timed out: the provider to retry it on (None when there is no other).
        The third fallback in this Job moves its first provider to the next one."""
        if len(self.order) < 2:
            return None
        with self._lock:
            self.fallbacks_total += 1
            self._since_switch += 1
            if self._since_switch >= FALLBACKS_TO_SWITCH:
                self._first, self._since_switch = (self._first + 1) % len(self.order), 0
            return self.order[(self.order.index(failed) + 1) % len(self.order)]


_POOLS: dict[tuple[str, ...], Pool] = {}
_POOLS_LOCK = threading.Lock()


def shared_pool(names: list[str]) -> Pool:
    """One pool per provider order in this process: all trials of an evaluation Job share it."""
    with _POOLS_LOCK:
        return _POOLS.setdefault(tuple(names), Pool(names))
