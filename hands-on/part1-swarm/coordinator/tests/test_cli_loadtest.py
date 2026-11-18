from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from conftest import GRID, PRICES
from swarm_coordinator.auth import FakeAuth, NoAuth, WhoamiAuth
from swarm_coordinator.cli import build, parse
from swarm_coordinator.loadtest import merge, percentile, run_loadtest
from swarm_coordinator.sink import BucketSink, NullSink
from test_sink import FakeBucketApi


@pytest.fixture
def files(tmp_path):
    g, p = tmp_path / "grid.json", tmp_path / "prices.json"
    g.write_text(json.dumps(GRID))
    p.write_text(json.dumps(PRICES))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"results": []}))
    return ["--grid", str(g), "--prices", str(p), "--reference", str(ref)]


def test_replay_mode_is_offline_and_read_only(files, monkeypatch):
    monkeypatch.delenv("SWARM_BUCKET", raising=False)
    c = build(parse(["--replay", "--replay-file", files[5], *files]), env={})
    assert isinstance(c.auth, NoAuth) and isinstance(c.sink, NullSink) and c.replayer is not None and c.mode == "replay"


def test_replay_server_streams_results_once_started(files, tmp_path):
    import time as _time
    from fastapi.testclient import TestClient

    from conftest import make_result
    ref = tmp_path / "rec.json"
    ref.write_text(json.dumps({"results": [make_result(c) for c in ["opencode__t1__0", "opencode__t2__0", "openclaw__t1__0"]]}))
    c = build(parse(["--replay", "--replay-file", str(ref), "--replay-minutes", "0.01", *files]), env={})
    with TestClient(c.app) as client:  # runs the lifespan: the replay starts with the server
        for _ in range(40):
            if client.get("/healthz").json()["version"] >= 6:
                break
            _time.sleep(0.1)
        assert len(c.store.results()) == 3
        assert client.post("/api/claim", headers={"Authorization": "Bearer x"}).status_code == 503  # read-only


def test_live_mode_with_a_bucket_restores_from_it(files):
    api = FakeBucketApi()
    api.files["results/opencode__t1__0.json"] = json.dumps({"cell_id": "opencode__t1__0", "harness": "opencode", "task": "t1", "trial": 0,
                                                             "passed": True, "hf_user": "alice", "received_at": "2026-11-30T15:00:00+00:00"}).encode()
    c = build(parse(files), env={"SWARM_BUCKET": "agent-harness-workshop/aidev-swarm-results", "HF_TOKEN": "x"}, hub_api=api)
    assert isinstance(c.auth, WhoamiAuth) and isinstance(c.sink, BucketSink)
    assert [r["cell_id"] for r in c.store.results()] == ["opencode__t1__0"]


def test_live_mode_without_a_bucket_runs_unpersisted(files):
    c = build(parse(files), env={})
    assert isinstance(c.sink, NullSink) and c.mode == "live"


def test_fake_auth_is_refused_on_a_space(files):
    with pytest.raises(SystemExit):
        build(parse(["--fake-auth", *files]), env={"SPACE_ID": "agent-harness-workshop/aidev-swarm"})
    assert isinstance(build(parse(["--fake-auth", *files]), env={}).auth, FakeAuth)


def test_percentile():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5.0], 50) == 5.0


def test_merge_finds_a_cell_claimed_in_two_processes():
    a = {"lat": {"claim": [1.0]}, "claims": {"x__t__0": ["u1"]}, "errors": [], "wall": 2.0}
    b = {"lat": {"claim": [3.0]}, "claims": {"x__t__0": ["u9"], "y__t__0": ["u9"]}, "errors": ["e"], "wall": 3.0}
    m = merge([a, b])
    assert m["claims"] == {"x__t__0": ["u1", "u9"], "y__t__0": ["u9"]}
    assert m["lat"]["claim"] == [1.0, 3.0] and m["errors"] == ["e"] and m["wall"] == 3.0


def test_loadtest_fills_the_grid_without_double_claims(files):
    big = {**GRID, "tasks": [f"t{i}" for i in range(10)], "trials_per_cell": 3}
    grid_file = files[1]
    with open(grid_file, "w") as f:
        json.dump(big, f)
    c = build(parse(["--fake-auth", *files]), env={})
    transport = httpx.ASGITransport(app=c.app)
    report = asyncio.run(run_loadtest("http://test", n_runners=25, transport=transport, viewers=3))
    assert report["cells_total"] == report["cells_done"] == 90
    assert report["double_claimed"] == [] and report["server_max_claims_per_cell"] == 1
    assert report["errors"] == 0
    assert report["latency_ms"]["claim"]["n"] == 90 + 25  # one 204 per runner at the end
    assert report["latency_ms"]["submit"]["p95"] >= 0


# ---- Part 2: the evolve board in the same process -----------------------------------------------------

def test_live_mode_restores_evolve_events_from_the_bucket_and_shows_the_recorded_backdrop(files):
    api = FakeBucketApi()
    api.files["evolve/alice/alice-r1/seed-000.json"] = json.dumps({"schema": "aidev-evolve/1", "run_id": "alice-r1", "kind": "seed", "iteration": 0,
                                                                   "val_score": 0.4, "search_score": 0.0, "spend_usd": 0.1,
                                                                   "user": "alice", "received_at": 1.0}).encode()
    c = build(parse(files), env={"SWARM_BUCKET": "agent-harness-workshop/aidev-swarm-results", "HF_TOKEN": "x"}, hub_api=api)
    assert [e["user"] for e in c.evolve_store.events()] == ["alice"]
    assert c.evolve_replayer is None
    from fastapi.testclient import TestClient
    st = TestClient(c.app).get("/api/evolve/state").json()
    assert [r["label"] for r in st["reference"]] == ["recorded B1", "recorded B2"]
    assert st["totals"]["attendees"] == 1


def test_replay_mode_streams_the_recorded_evolve_runs(files):
    import time as _time
    from fastapi.testclient import TestClient

    c = build(parse(["--replay", "--replay-minutes", "0.01", *files]), env={})
    with TestClient(c.app) as client:
        for _ in range(40):
            if len(c.evolve_store.events()) == 20:
                break
            _time.sleep(0.1)
        assert len(c.evolve_store.events()) == 20  # B1 and B2: seed + 8 iterations + held-out each
        assert client.post("/api/evolve/submit", json={}, headers={"Authorization": "Bearer x"}).status_code == 503  # read-only


def test_evolve_loadtest_posts_every_attendees_events_and_the_board_counts_them(files):
    from swarm_coordinator.loadtest import run_evolve_loadtest

    c = build(parse(["--fake-auth", *files]), env={})
    transport = httpx.ASGITransport(app=c.app)
    report = asyncio.run(run_evolve_loadtest("http://test", attendees=30, interval_s=0.05, iterations=3, viewers=2, transport=transport))
    assert report["errors"] == 0 and report["rate_limited"] == 0
    assert report["latency_ms"]["submit"]["n"] == 30 * 4  # seed + 3 iterations each
    assert report["board"]["attendees"] == 30 and report["board"]["iterations"] == 90
    assert report["latency_ms"]["state"]["n"] > 0
