"""Which GLM-5.3-Flash provider serves an attendee (config/providers.json, PROTOCOL.md "Providers").

Weighted rendezvous hashing: every (user, provider) pair gets a pseudo-random u in (0, 1) from SHA-256, and the
providers are ordered by -ln(u) / weight, smallest first. The first is the attendee's provider, the rest are the
fallback in order. Each provider is first for a share of attendees equal to its share of the weights, an attendee
always gets the same order, and dropping a provider moves only the attendees it served. Part 2's client implements
the same function; PROTOCOL.md lists TEST_VECTORS so both can check they agree.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib import resources
from typing import Any

# Cross-implementation check values for the 3:1 novita/zai-org list (PROTOCOL.md "Providers").
TEST_VECTORS = {"alice": ["novita", "zai-org"], "dave": ["zai-org", "novita"], "JoelNiklaus": ["novita", "zai-org"]}


def load() -> list[dict[str, Any]]:
    """The packaged allowlist (a copy of config/providers.json)."""
    return json.loads(resources.files("aidev_swarm").joinpath("data", "providers.json").read_text())["providers"]


def _u(user: str, provider: str) -> float:
    h = hashlib.sha256(f"{user.lower()}\n{provider}".encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2**64


def order(user: str, providers: list[dict[str, Any]]) -> list[str]:
    """The attendee's providers, preferred first. Providers with weight 0 are left out."""
    live = [p for p in providers if p["weight"] > 0]
    return [p["name"] for p in sorted(live, key=lambda p: (-math.log(_u(user, p["name"])) / p["weight"], p["name"]))]


def provider_errors(meter_log: str) -> int:
    """Calls the trial's provider answered with 429 or 5xx, from the in-sandbox meter's log (agent/hlb-meter.jsonl):
    the final status, or `fallback` when the meter sent the call on to the reference after the provider's 429."""
    n = 0
    for line in meter_log.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict) or "status" not in rec:
            continue
        status = rec.get("fallback") if isinstance(rec.get("fallback"), int) else rec.get("status")
        n += isinstance(status, int) and (status == 429 or status >= 500)
    return n
