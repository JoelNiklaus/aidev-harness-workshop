"""Build the Result record from a finished Harbor trial folder, and tell an infrastructure error from a fail.

Token counts come from the in-sandbox meter's log (`agent/hlb-meter.jsonl`, summarized by the Harness
Leaderboard's `scoring.meter_summary`), never from what a harness reports. Cost is tokens x the list
price in `data/prices.json`, cached input at its own rate, so every attendee and the slides use the same numbers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hlb.scoring import meter_summary

from aidev_lb.providers import call_counts

METER_LOG = Path("agent") / "hlb-meter.jsonl"


def cost_usd(n_input: int, n_cached: int, n_output: int, price: dict[str, float]) -> float:
    cached = min(n_cached, n_input)
    return ((n_input - cached) * price["input"] + cached * price["cached_input"] + n_output * price["output"]) / 1e6


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
    return round((parse(end) - parse(start)).total_seconds(), 3)


def _error(result: dict[str, Any]) -> str | None:
    exc = result.get("exception_info") or {}
    if not exc:
        return None
    return f"{exc.get('exception_type')}: {(exc.get('exception_message') or '')[-300:]}".strip()


def meter_text(trial_dir: Path) -> str:
    path = Path(trial_dir) / METER_LOG
    return path.read_text() if path.exists() else ""


def infra_error(result: dict[str, Any], meter_log: str) -> str | None:
    """Why a trial says nothing about the harness, or None for a trial that counts. A trial that did not pass is an
    infrastructure error when it ended with an error before any model call (sandbox, image, harness install), or when
    the provider answered none of its model calls (refused or rate-limited every one)."""
    if result["passed"]:
        return None
    counts = call_counts(meter_log)
    if counts["provider_errors"] and not counts["answered"]:
        return f"the provider refused every model call ({counts['provider_errors']} of {result['n_calls']})"
    if result["error"] and not result["n_calls"]:
        return f"ended before any model call: {result['error']}"
    return None


def build_result(cell: dict[str, Any], trial_dir: Path, price: dict[str, float], *, hf_user: str, runner_version: str,
                 provider: str, sandbox_id: str | None = None, trajectory_url: str | None = None) -> dict[str, Any]:
    """`provider`: the GLM-5.3-Flash provider the trial was routed to; `n_provider_errors`: its calls that ended in a
    provider error (providers.is_provider_error). `source` is "measured": the attendee's own run."""
    trial_dir = Path(trial_dir)
    try:
        result = json.loads((trial_dir / "result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        result = {"exception_info": {"exception_type": "NoTrialResult", "exception_message": f"{type(e).__name__} reading result.json"}}
    meter = meter_text(trial_dir)
    m = meter_summary(meter) if meter else {"input": 0, "cached": 0, "output": 0, "n_calls": 0}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = float(rewards.get("reward", 0.0))
    return {
        "cell_id": cell["cell_id"],
        "harness": cell["harness"],
        "harness_version": (result.get("agent_info") or {}).get("version"),
        "task": cell["task"],
        "trial": cell["trial"],
        "model": cell["model"],
        "reward": reward,
        "passed": reward >= 1.0,
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "wall_s": _seconds(result.get("started_at"), result.get("finished_at")),
        "n_input_tokens": m["input"],
        "n_cached_tokens": m["cached"],
        "n_output_tokens": m["output"],
        "n_calls": m["n_calls"],
        "cost_usd": cost_usd(m["input"], m["cached"], m["output"], price),
        "error": _error(result),
        "hf_user": hf_user,
        "runner_version": runner_version,
        "sandbox_id": sandbox_id,
        "trajectory_url": trajectory_url,
        "provider": provider,
        "n_provider_errors": call_counts(meter)["provider_errors"],
        "source": "measured",
    }
