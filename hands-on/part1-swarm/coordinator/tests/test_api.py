from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from conftest import make_result
from swarm_coordinator.app import create_app
from swarm_coordinator.auth import AuthUnavailable, Unauthorized


class TokenAuth:
    """Token 'tok-<name>' is user <name>; 'down' simulates a Hub outage."""

    def user(self, authorization):
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token == "down":
            raise AuthUnavailable("hub down")
        if not token.startswith("tok-"):
            raise Unauthorized("bad token")
        return token[4:]


class Mono:
    t = 0.0

    def __call__(self):
        return self.t


RUNNER_UA = "aidev-swarm/0.4.1"


def H(user, ua=RUNNER_UA):
    return {"Authorization": f"Bearer tok-{user}", "User-Agent": ua}


@pytest.fixture
def mono():
    return Mono()


@pytest.fixture
def client(store, mono):
    ref = [{**make_result("pi__t1__0", user="harness-leaderboard"), "cost_usd": 0.01}]
    app = create_app(store, TokenAuth(), reference=ref, meta={}, mode="live", mono=mono)
    return TestClient(app)


def test_claim_needs_auth(client):
    assert client.post("/api/claim").status_code == 401
    assert client.post("/api/claim", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_hub_outage_is_503(client):
    assert client.post("/api/claim", headers={"Authorization": "Bearer down"}).status_code == 503


def test_claim_returns_a_cell(client):
    r = client.post("/api/claim", headers=H("alice"))
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"cell_id", "harness", "task", "trial", "model", "lease_expires_at", "run_config"}
    assert body["cell_id"] == "opencode__t1__0"


def test_second_claim_is_409_with_held_cells(client):
    client.post("/api/claim", headers=H("alice"))
    r = client.post("/api/claim", headers=H("alice"))
    assert r.status_code == 409 and r.json()["held"] == ["opencode__t1__0"]


def test_claim_n2_returns_a_list(client):
    r = client.post("/api/claim?n=2", headers=H("alice"))
    assert r.status_code == 200 and [c["cell_id"] for c in r.json()] == ["opencode__t1__0", "opencode__t2__0"]


def test_claim_n3_is_422(client):
    assert client.post("/api/claim?n=3", headers=H("alice")).status_code == 422


def test_claim_204_when_nothing_is_left(client):
    for i in range(12):
        assert client.post("/api/claim", headers=H(f"u{i}")).status_code == 200
    r = client.post("/api/claim", headers=H("late"))
    assert r.status_code == 204 and r.content == b""


def test_claim_204_says_when_to_ask_again_while_leased_cells_may_still_come_back(client, store):
    for i in range(12):
        assert client.post("/api/claim", headers=H(f"u{i}")).status_code == 200
    r = client.post("/api/claim", headers=H("late"))
    assert r.status_code == 204 and int(r.headers["retry-after"]) > 0
    for c in store.snapshot()["leases"]:
        store.submit(c["cell_id"], c["user"], make_result(c["cell_id"], user=c["user"]))
    r = client.post("/api/claim", headers=H("late"))
    assert r.status_code == 204 and "retry-after" not in r.headers  # the grid is full: stop


@pytest.mark.parametrize("ua", ["python-httpx/0.28.1", "aidev-swarm/0.3.1", "aidev-swarm/0.4.0", "testclient", ""])
def test_a_claim_from_a_runner_older_than_the_short_lease_protocol_is_426(client, ua):
    """0.3.1 beats every 2 min against a 90 s lease: it would lose its cells. 0.4.0 sends every trial to zai-org and
    records no provider (PROTOCOL.md "Providers"). Neither may claim."""
    r = client.post("/api/claim", headers=H("alice", ua=ua))
    assert r.status_code == 426 and "aidev-swarm" in r.json()["detail"] and "/join" in r.json()["detail"]
    assert client.get("/api/state").json()["counts"]["running"] == 0


@pytest.mark.parametrize("ua", ["aidev-swarm/0.4.1", "aidev-swarm/0.4.12", "aidev-swarm/1.0.0 (job)"])
def test_current_runners_may_claim(client, ua):
    assert client.post("/api/claim", headers=H("alice", ua=ua)).status_code == 200


def test_full_cycle_heartbeat_submit(client):
    c = client.post("/api/claim", headers=H("alice")).json()
    cid = c["cell_id"]
    hb = client.post(f"/api/heartbeat/{cid}", headers=H("alice"))
    assert hb.status_code == 200 and hb.json()["lease_expires_at"]
    assert client.post(f"/api/heartbeat/{cid}", headers=H("bob")).status_code == 409
    assert client.post(f"/api/submit/{cid}", headers=H("bob"), json=make_result(cid)).status_code == 409
    r = client.post(f"/api/submit/{cid}", headers=H("alice"), json=make_result(cid))
    assert r.status_code == 200 and r.json()["ok"] is True


def test_submit_validation_is_422(client):
    cid = client.post("/api/claim", headers=H("alice")).json()["cell_id"]
    assert client.post(f"/api/submit/{cid}", headers=H("alice"), json={"cell_id": cid}).status_code == 422
    assert client.post(f"/api/submit/{cid}", headers=H("alice"), content=b"not json").status_code == 422


def test_unknown_cell_is_404(client):
    assert client.post("/api/heartbeat/nope__x__0", headers=H("alice")).status_code == 404


def test_url_encoded_cell_id_works(client):
    cid = client.post("/api/claim", headers=H("alice")).json()["cell_id"]
    assert client.post("/api/release/" + cid.replace("_", "%5F"), headers=H("alice")).status_code == 200
    assert client.post("/api/claim", headers=H("bob")).json()["cell_id"] == cid


def test_state_is_public_cached_and_etagged(client, mono):
    s = client.get("/api/state")
    assert s.status_code == 200 and s.json()["counts"]["cells_total"] == 12
    etag = s.headers["etag"]
    assert client.get("/api/state", headers={"If-None-Match": etag}).status_code == 304
    client.post("/api/claim", headers=H("alice"))
    # within the 2 s cache window the payload stays the same
    assert client.get("/api/state").json()["counts"]["running"] == 0
    mono.t += 2.5
    assert client.get("/api/state").json()["counts"]["running"] == 1


def test_board_page_is_self_contained(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    html = r.text
    assert 'fetch("api/state"' in html  # relative, so the page also works under a path prefix
    # offline: no external scripts, styles or fonts
    assert not re.search(r"""(src|href)=["']https?://""", html)
    assert "@import" not in html and "fonts.googleapis" not in html
    assert len(html.encode()) < 60_000


def test_board_page_embeds_the_current_state(client):
    # one round trip to the first full draw: the private Space's proxy costs ~0.3 s per request
    client.post("/api/claim", headers=H("alice"))
    html = client.get("/").text
    m = re.search(r'<script id="boot" type="application/json">(.*?)</script>', html, re.S)
    assert m, "no embedded state"
    import json as _json
    boot = _json.loads(m.group(1))
    assert boot["counts"]["cells_total"] == 12 and boot["counts"]["running"] == 1


def test_embedded_state_cannot_break_out_of_its_script_tag(store, mono):
    evil = "</script><script>alert(1)</script>"
    c = store.claim(evil)[0]
    store.submit(c["cell_id"], evil, make_result(c["cell_id"]))
    app = create_app(store, TokenAuth(), reference=[], meta={}, mode="live", mono=mono)
    html = TestClient(app).get("/").text
    assert "alert(1)</script>" not in html
    # the boot block ends at its own closing tag and still carries the exact name
    m = re.search(r'<script id="boot" type="application/json">(.*?)</script>', html, re.S)
    import json as _json
    assert _json.loads(m.group(1))["feed"][0]["hf_user"] == evil


def test_board_polls_gently_by_default_and_backs_off(client):
    html = client.get("/").text
    # attendees' browsers share the venue IP with the runners: 10 s by default, ?every= for the projector,
    # and doubling back-off up to 60 s on errors and 429s from the HF edge
    assert "every" in html and "URLSearchParams" in html
    assert "10000" in html and "60000" in html


def test_healthz(client):
    assert client.get("/healthz").json()["ok"] is True


def test_the_current_runner_version_may_claim():
    """Drift guard: the runner in this repo (runner/src/aidev_swarm/__init__.py) passes the coordinator's version gate."""
    from pathlib import Path

    from swarm_coordinator.app import runner_ok

    init = Path(__file__).resolve().parents[2] / "runner" / "src" / "aidev_swarm" / "__init__.py"
    version = re.search(r'__version__ = "([^"]+)"', init.read_text()).group(1)
    assert runner_ok(f"aidev-swarm/{version}")
