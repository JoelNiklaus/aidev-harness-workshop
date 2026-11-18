"""Build the swarm's Result record (PROTOCOL.md) from a finished Harbor trial folder.

Token counts come from the in-sandbox meter's log (`agent/hlb-meter.jsonl`, summarized by the Harness
Leaderboard's `scoring.meter_summary`), never from what a harness reports. Cost is tokens x the list
price in `config/prices.json`, cached input at its own rate, so board and slides use the same numbers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hlb.scoring import meter_summary

from aidev_swarm.providers import provider_errors

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


def build_result(claim: dict[str, Any], trial_dir: Path, price: dict[str, float], *, hf_user: str, runner_version: str,
                 provider: str, sandbox_id: str | None = None, trajectory_url: str | None = None) -> dict[str, Any]:
    """`provider`: the GLM-5.3-Flash provider the trial was routed to (PROTOCOL.md "Providers"); `n_provider_errors`:
    its calls answered 429/5xx (the meter sends a 429'd call of a non-reference provider on to zai-org)."""
    trial_dir = Path(trial_dir)
    try:
        result = json.loads((trial_dir / "result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        result = {"exception_info": {"exception_type": "NoTrialResult", "exception_message": f"{type(e).__name__} reading result.json"}}
    meter_path = trial_dir / METER_LOG
    meter_text = meter_path.read_text() if meter_path.exists() else ""
    m = meter_summary(meter_text) if meter_text else {"input": 0, "cached": 0, "output": 0, "n_calls": 0}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = float(rewards.get("reward", 0.0))
    return {
        "cell_id": claim["cell_id"],
        "harness": claim["harness"],
        "harness_version": (result.get("agent_info") or {}).get("version"),
        "task": claim["task"],
        "trial": claim["trial"],
        "model": claim["model"],
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
        "n_provider_errors": provider_errors(meter_text),
    }
