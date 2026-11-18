from __future__ import annotations

import pytest

from conftest import make_result
from swarm_coordinator.state import aggregate, build_state, wilson

META = {"opencode": {"display": "OpenCode", "color": "#3E8E5E"}, "openclaw": {"display": "OpenClaw", "color": "#EF3E36"}}


def test_wilson_interval():
    lo, hi = wilson(7, 10)
    assert lo == pytest.approx(0.3968, abs=1e-3) and hi == pytest.approx(0.8922, abs=1e-3)
    assert wilson(0, 0) == (None, None)


def test_aggregate_per_harness():
    rs = [
        {**make_result("opencode__t1__0", passed=True), "cost_usd": 0.02},
        {**make_result("opencode__t2__0", passed=False), "cost_usd": 0.04},
    ]
    a = aggregate(rs)["opencode"]
    assert a["n"] == 2 and a["n_pass"] == 1 and a["pass_rate"] == 0.5
    assert a["ci"][0] < 0.5 < a["ci"][1]
    assert a["cost_per_trial"] == pytest.approx(0.03)
    assert a["cost_per_pass"] == pytest.approx(0.06)
    assert a["tokens_per_trial"] == pytest.approx(1_010_000)


def test_cost_per_pass_is_none_without_passes():
    a = aggregate([make_result("opencode__t1__0", passed=False)])["opencode"]
    assert a["cost_per_pass"] is None


def test_state_shows_progress_grid_feed_and_unreached_harnesses(store, clock, grid):
    a = store.claim("alice")[0]
    store.submit(a["cell_id"], "alice", make_result(a["cell_id"], passed=True))
    clock.advance(5)
    b = store.claim("bob")[0]
    store.submit(b["cell_id"], "bob", make_result(b["cell_id"], passed=False))
    store.claim("carol")  # opencode__t1__1 running
    ref = [{**make_result("openclaw__t1__0", user="harness-leaderboard"), "cost_usd": 0.05}]
    s = build_state(grid, store.snapshot(), ref, META, mode="live")

    assert s["counts"] == {"cells_total": 12, "done": 2, "passed": 1, "running": 1, "contributors": 3, "max_claims_per_cell": 1, "by_provider": {"unknown": 2}}
    # grid: one string per harness, task-major blocks of trials: t1 trials then t2 trials
    assert s["grid"]["tasks"] == ["t1", "t2"] and s["grid"]["trials"] == 2
    assert s["grid"]["rows"]["opencode"] == "pcfo"
    assert s["grid"]["rows"]["openclaw"] == "oooo"
    hs = {h["harness"]: h for h in s["harnesses"]}
    assert [h["harness"] for h in s["harnesses"]] == ["opencode", "openclaw", "pi"]  # grid order
    assert hs["opencode"]["reached"] and hs["opencode"]["room"]["n"] == 2
    assert not hs["openclaw"]["reached"] and hs["openclaw"]["room"] is None
    assert hs["openclaw"]["reference"]["n"] == 1  # reference layer for every harness
    assert hs["opencode"]["display"] == "OpenCode" and hs["pi"]["display"] == "pi"  # fallback display name
    feed = s["feed"]
    assert [f["hf_user"] for f in feed] == ["bob", "alice"]  # newest first
    assert feed[0]["passed"] is False and feed[0]["harness"] == "opencode"
    assert s["mode"] == "live" and s["model"] == grid.model


def test_running_harness_counts_as_reached(store, grid):
    store.claim("alice")
    s = build_state(grid, store.snapshot(), [], META, mode="live")
    assert s["harnesses"][0]["reached"] is True and s["harnesses"][0]["room"] is None


def test_results_outside_the_grid_do_not_count(store, grid):
    snap = store.snapshot()
    snap["results"] = [make_result("gone__t9__0")]
    s = build_state(grid, snap, [], META, mode="live")
    assert s["counts"]["done"] == 0


def test_state_counts_results_per_provider(store, grid):
    # PROTOCOL.md "Providers": each result names its provider (0.4.0 runners send none)
    for user, provider in (("alice", "novita"), ("bob", "zai-org"), ("carol", None)):
        c = store.claim(user)[0]
        extra = {"provider": provider, "n_provider_errors": 0} if provider else {}
        store.submit(c["cell_id"], user, make_result(c["cell_id"], user=user, **extra))
    s = build_state(grid, store.snapshot(), [], META, mode="live")
    assert s["counts"]["by_provider"] == {"novita": 1, "zai-org": 1, "unknown": 1}
