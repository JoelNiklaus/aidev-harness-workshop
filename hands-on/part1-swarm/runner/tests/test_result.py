"""The Result record (PROTOCOL.md) built from a finished Harbor trial folder."""

import json
from pathlib import Path

import pytest

from aidev_swarm import result as R

FIX = Path(__file__).parent / "fixtures" / "trial_pass"
PRICE = {"input": 0.15, "cached_input": 0.03, "output": 0.5}
CLAIM = {"cell_id": "pi|nginx-request-logging|0", "harness": "pi", "task": "nginx-request-logging", "trial": 0, "model": "zai-org/GLM-5.3-Flash"}
KEYS = {"cell_id", "harness", "harness_version", "task", "trial", "model", "reward", "passed", "started_at", "finished_at", "wall_s", "n_input_tokens",
        "n_cached_tokens", "n_output_tokens", "n_calls", "cost_usd", "error", "hf_user", "runner_version", "sandbox_id", "trajectory_url"}


def build(trial_dir=FIX, **kw):
    return R.build_result(CLAIM, trial_dir, PRICE, hf_user="alice", runner_version="0.1.0", **kw)


def test_result_has_exactly_the_protocol_fields():
    assert set(build()) == KEYS


def test_passed_trial_from_the_real_smoke():
    r = build()
    meter = [json.loads(line) for line in (FIX / "agent" / "hlb-meter.jsonl").read_text().splitlines()]
    calls = [m for m in meter if m.get("path") == "/v1/chat/completions"]
    assert r["passed"] is True and r["reward"] == 1.0 and r["error"] is None
    assert r["harness_version"] == "1.2.0"
    assert r["n_calls"] == len(calls) == 8
    assert r["n_input_tokens"] == sum(c["input"] for c in calls)
    assert r["n_output_tokens"] == sum(c["output"] for c in calls)
    assert r["n_cached_tokens"] == sum(c.get("cached") or 0 for c in calls)
    assert r["wall_s"] == pytest.approx(87.07, abs=0.01)
    assert r["started_at"].startswith("2026-11-17T09:48:20")
    assert r["hf_user"] == "alice" and r["cell_id"] == CLAIM["cell_id"]


def test_cost_is_tokens_times_list_price_with_cached_input_at_its_rate():
    r = build()
    uncached = r["n_input_tokens"] - r["n_cached_tokens"]
    want = (uncached * 0.15 + r["n_cached_tokens"] * 0.03 + r["n_output_tokens"] * 0.5) / 1e6
    assert r["cost_usd"] == pytest.approx(want, rel=1e-9)
    assert 0.0005 < r["cost_usd"] < 0.01


def test_failed_reward_is_not_passed(tmp_path):
    d = copy_fixture(tmp_path, reward=0.0)
    r = build(d)
    assert r["reward"] == 0.0 and r["passed"] is False and r["error"] is None


def test_exception_without_verifier_reports_error_and_zero_reward(tmp_path):
    d = copy_fixture(tmp_path, reward=None, exception={"exception_type": "AgentSetupTimeoutError", "exception_message": "install took too long"})
    r = build(d)
    assert r["passed"] is False and r["reward"] == 0.0
    assert r["error"].startswith("AgentSetupTimeoutError")


def test_missing_meter_log_gives_zero_tokens_and_cost(tmp_path):
    d = copy_fixture(tmp_path)
    (d / "agent" / "hlb-meter.jsonl").unlink()
    r = build(d)
    assert (r["n_input_tokens"], r["n_output_tokens"], r["n_calls"], r["cost_usd"]) == (0, 0, 0, 0.0)


def test_missing_result_json_is_an_error_result(tmp_path):
    r = build(tmp_path)
    assert r["passed"] is False and r["error"] and r["harness"] == "pi" and set(r) == KEYS


def copy_fixture(tmp_path, reward=1.0, exception=None):
    d = tmp_path / "trial"
    (d / "agent").mkdir(parents=True)
    res = json.loads((FIX / "result.json").read_text())
    res["verifier_result"] = None if reward is None else {"rewards": {"reward": reward}}
    res["exception_info"] = exception
    (d / "result.json").write_text(json.dumps(res))
    (d / "agent" / "hlb-meter.jsonl").write_text((FIX / "agent" / "hlb-meter.jsonl").read_text())
    return d
