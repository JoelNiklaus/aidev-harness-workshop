"""Which GLM-5.3-Flash provider serves an attendee, and which of a provider's answers count as its failure.

The list is the packaged data/providers.json. Weighted rendezvous hashing: every (user, provider) pair gets a
pseudo-random u in (0, 1) from SHA-256, and the providers are ordered by -ln(u) / weight, smallest first. The first is
the attendee's provider, the rest are the fallback in order. Each provider is first for a share of attendees equal to
its share of the weights, an attendee always gets the same order, and dropping a provider moves only the attendees it
served.

A provider error is a rate limit or outage (429, 5xx) or a refusal (400 "not supported by provider", 401/403, 404):
the trial or check moves to the next provider, and a trial whose every model call ended that way is an infrastructure
error, never a fail (result.infra_error).
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
    """The packaged provider list: [{"name", "weight", ...}]."""
    return json.loads(resources.files("aidev_lb").joinpath("data", "providers.json").read_text())["providers"]


def _u(user: str, provider: str) -> float:
    h = hashlib.sha256(f"{user.lower()}\n{provider}".encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2**64


def order(user: str, providers: list[dict[str, Any]]) -> list[str]:
    """The attendee's providers, preferred first. Providers with weight 0 are left out."""
    live = [p for p in providers if p["weight"] > 0]
    return [p["name"] for p in sorted(live, key=lambda p: (-math.log(_u(user, p["name"])) / p["weight"], p["name"]))]


def is_provider_error(status: int | None, body: str = "") -> bool:
    """Whether an HTTP answer to a model call is the provider's failure: 429/5xx, or a refusal of the model or the
    route (measured 25 Nov 2026: HTTP 400 "The requested model ... is not supported by provider 'novita'"). A 402
    (no credits) and any other 400 (a request the harness built wrong) are not."""
    if not isinstance(status, int):
        return False
    if status == 429 or status >= 500 or status in (401, 403, 404):
        return True
    return status == 400 and "not supported" in body.lower()


def call_counts(meter_log: str) -> dict[str, int]:
    """The trial's model calls from the in-sandbox meter's log (agent/hlb-meter.jsonl): `answered` with 200, and
    `provider_errors` (is_provider_error on the final status, or on `fallback` when the meter sent the call on to the
    next provider after a 429: such a call counts in both)."""
    answered = errors = 0
    for line in meter_log.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict) or "status" not in rec or rec.get("refused"):
            continue
        answered += rec["status"] == 200
        errors += is_provider_error(rec.get("fallback"), "") or is_provider_error(rec["status"], str(rec.get("error_body") or ""))
    return {"answered": answered, "provider_errors": errors}
