"""The event lines a Job prints on its log, and the plain lines the laptop shows for them.

A Job has no other channel to the laptop than its log, so every event is one line `AIDEV-LB {json}` between the
installer's and Harbor's own output. Events: job_started, provider, sandbox_started, harness_installed,
agent_finished, retry, result (a full Result record), infra, store_error, job_done.
"""

from __future__ import annotations

import json
from typing import Any

PREFIX = "AIDEV-LB "
PHASE_TEXT = {"sandbox_started": "sandbox started", "harness_installed": "harness installed", "agent_finished": "agent finished"}


def line(event: str, **fields: Any) -> str:
    return PREFIX + json.dumps({"event": event, **fields}, separators=(",", ":"))


def parse(text: str) -> dict[str, Any] | None:
    """The event of a log line, or None for any other line."""
    at = text.find(PREFIX)
    if at < 0:
        return None
    try:
        ev = json.loads(text[at + len(PREFIX):])
    except ValueError:
        return None
    return ev if isinstance(ev, dict) and isinstance(ev.get("event"), str) else None


def plain(ev: dict[str, Any]) -> str | None:
    """One line for the attendee, or None for an event that needs none."""
    kind = ev["event"]
    if kind in PHASE_TEXT:
        return f"{ev['harness']}  {ev['task']}  {PHASE_TEXT[kind]}"
    if kind == "result":
        r = ev["result"]
        return f"{r['harness']}  {r['task']}  {'passed' if r['passed'] else 'failed'}  {r['wall_s'] or 0:.0f} s  ${r['cost_usd']:.4f}"
    if kind == "infra":
        return f"{ev['harness']}  {ev['task']}  infrastructure error, not scored: {ev['error']}"
    if kind == "retry":
        return f"{ev['harness']}  {ev['task']}  again on {ev['provider']}: {ev['why']}"
    if kind == "provider":
        skipped = "; ".join(f"{s['provider']} skipped: {s['why']}" for s in ev.get("skipped") or [])
        return f"{ev['harness']}  model calls go to {ev['provider']}" + (f" ({skipped})" if skipped else "")
    if kind == "store_error":
        return f"{ev['harness']}  could not write to the bucket {ev['bucket']} ({ev['error']}); the results still arrive here"
    if kind == "job_started":
        return f"{ev['harness']}  Job running: {len(ev['tasks'])} trials start, each in its own sandbox"
    return None
