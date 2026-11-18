"""The /evolve HTTP surface: submit (auth, limits, idempotency), state, diff, page."""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from swarm_coordinator import evolve
from swarm_coordinator.app import create_app

from conftest import Clock
from test_api import H, Mono, TokenAuth
from test_evolve import heldout, it, seed


@pytest.fixture
def ev_clock():
    return Clock()


@pytest.fixture
def ev_store(ev_clock):
    return evolve.EvolveStore(clock=ev_clock)


@pytest.fixture
def limiter(ev_clock):
    return evolve.RateLimiter(per_minute=6, burst=4, clock=ev_clock)


@pytest.fixture
def mono():
    return Mono()


@pytest.fixture
def client(store, ev_store, limiter, mono):
    ref = [{"label": "recorded B1", "curve": [[0, 0.42], [1, 0.88]]}]
    app = create_app(store, TokenAuth(), reference=[], meta={}, mode="live", mono=mono,
                     evolve_store=ev_store, evolve_reference=ref, evolve_limiter=limiter)
    return TestClient(app)


def post(client, event, user="alice", **kw):
    return client.post("/api/evolve/submit", json=event, headers=H(user), **kw)


def test_submit_needs_a_known_token(client):
    assert client.post("/api/evolve/submit", json=seed()).status_code == 401
    assert client.post("/api/evolve/submit", json=seed(), headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.post("/api/evolve/submit", json=seed(), headers={"Authorization": "Bearer down"}).status_code == 503


def test_submit_stores_the_event_under_the_tokens_user(client, ev_store):
    r = post(client, seed(run_id="whatever-name"), user="alice")
    assert r.status_code == 200 and r.json()["ok"] is True
    (e,) = ev_store.events()
    assert e["user"] == "alice" and e["run_id"] == "whatever-name"


def test_submit_rejects_bad_bodies_with_422(client):
    assert client.post("/api/evolve/submit", content=b"{not json", headers=H("alice")).status_code == 422
    assert post(client, {**seed(), "kind": "nope"}).status_code == 422


def test_submit_refuses_oversized_bodies_with_413(client, ev_store):
    big = {**it(1), "diff": "x" * 7000, "padding": "y" * evolve.MAX_BODY_BYTES}
    r = post(client, big)
    assert r.status_code == 413
    assert ev_store.events() == []


def test_submit_rate_limits_per_user_with_retry_after(client):
    codes = [post(client, it(i)).status_code for i in range(1, 6)]
    assert codes == [200, 200, 200, 200, 429]
    r = post(client, it(6))
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert post(client, seed(), user="bob").status_code == 200  # other users are not affected


def test_retried_submit_is_idempotent(client, ev_store):
    for _ in range(3):
        assert post(client, it(1, accepted=True, best=0.8)).status_code == 200
    assert len(ev_store.events()) == 1


def test_state_reflects_submissions_after_the_cache_expires(client, mono):
    first = client.get("/api/evolve/state").json()
    assert first["totals"]["runs"] == 0 and first["reference"][0]["label"] == "recorded B1"
    post(client, seed())
    post(client, it(1, accepted=True, best=0.8))
    assert client.get("/api/evolve/state").json()["totals"]["runs"] == 0  # cached 2 s
    mono.t += 2.0
    st = client.get("/api/evolve/state").json()
    assert st["totals"] == {"attendees": 1, "runs": 1, "iterations": 1, "kept": 1, "spend_usd": 0.2}
    assert st["feed"][0]["title"] == "Show stderr to the model"


def test_state_answers_304_on_a_matching_etag(client):
    r = client.get("/api/evolve/state")
    assert client.get("/api/evolve/state", headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_diff_endpoint_serves_the_stored_diff_as_plain_text(client):
    post(client, it(3, accepted=True, best=0.8))
    r = client.get("/api/evolve/diff/alice/alice-r1/3")
    assert r.status_code == 200 and r.text == it(3)["diff"]
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/evolve/diff/alice/alice-r1/4").status_code == 404


def test_evolve_page_embeds_the_state_and_escapes_untrusted_text(client, mono):
    post(client, seed())
    post(client, it(1, accepted=True, best=0.8, title="</script><script>alert(1)</script>"))
    mono.t += 2.0
    html = client.get("/evolve").text
    boot = re.search(r'<script id="boot" type="application/json">(.*?)</script>', html, re.S).group(1)
    st = json.loads(boot)
    assert st["feed"][0]["title"] == "</script><script>alert(1)</script>"
    assert "</script><script>alert(1)" not in html  # inside the JSON block only "</" could end it


def test_heldout_event_shows_on_its_run(client, mono):
    post(client, seed())
    post(client, heldout())
    mono.t += 2.0
    (run,) = client.get("/api/evolve/state").json()["runs"]
    assert run["heldout"]["best"] == 0.75 and run["heldout"]["n_tasks"] == 8


def test_oversized_bodies_are_refused_before_the_token_is_checked(client):
    # the Content-Length check is the cheap one: no Hub call, no body read
    r = client.post("/api/evolve/submit", content=b"x" * (evolve.MAX_BODY_BYTES + 1), headers={"Authorization": "Bearer junk"})
    assert r.status_code == 413


def test_oversized_chunked_bodies_without_a_length_are_refused_too(client, ev_store):
    def chunks():
        yield b'{"pad": "' + b"x" * evolve.MAX_BODY_BYTES
        yield b'"}'
    r = client.post("/api/evolve/submit", content=chunks(), headers=H("alice"))
    assert r.status_code == 413
    assert ev_store.events() == []


def test_pages_and_state_are_gzipped_for_the_first_round_trip(client, mono):
    # an uncompressed 14-30 KB page needs extra round trips through the HF edge (measured +0.2 s from Europe)
    for i in range(1, 4):
        post(client, it(i, accepted=True, best=0.8))
    mono.t += 2.0
    for path in ("/evolve", "/", "/api/evolve/state"):
        r = client.get(path, headers={"Accept-Encoding": "gzip"})
        assert r.headers.get("content-encoding") == "gzip", path
