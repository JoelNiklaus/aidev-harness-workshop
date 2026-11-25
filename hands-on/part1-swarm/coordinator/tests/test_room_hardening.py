"""Freeze for the room (21 Nov): the board takes Part 2 events only from the frozen aidev-evolve; staging can stand in
for a room of attendees with one account (load-test names), prod never can."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from swarm_coordinator import app as app_mod
from swarm_coordinator import cli, deploy, paths
from swarm_coordinator.app import create_app

from test_api import H, TokenAuth
from test_deploy import FakeHub, token_file
from test_evolve import seed

EVOLVE_UA = "aidev-evolve/0.6.0"
AS = "X-Aidev-Loadtest-As"


@pytest.fixture
def client(store):
    return TestClient(create_app(store, TokenAuth(), reference=[], meta={}, loadtest_users=frozenset({"joel"})))


# ---- minimum aidev-evolve --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ua", [None, "python-httpx/0.28", "aidev-evolve/0.5.1", "aidev-evolve/0.5.3", "aidev-swarm/0.4.3"])
def test_an_event_from_an_evolve_client_older_than_the_frozen_one_is_426(client, ua):
    """Up to 0.5.1 the evaluation pinned leaderboard code that did not stop an agent at its timeout (and sent no version);
    a 0.5.3 run is the four-Job loop that ends after the session's ranking (30 min in a full room)."""
    headers = {k: v for k, v in H("alice", ua).items() if v is not None}
    r = client.post("/api/evolve/submit", json=seed(), headers=headers)
    assert r.status_code == 426 and "aidev-evolve >= 0.6.0" in r.json()["detail"] and "/evolve/join" in r.json()["detail"]
    assert client.get("/api/evolve/state").json()["totals"]["runs"] == 0


@pytest.mark.parametrize("ua", [EVOLVE_UA, "aidev-evolve/0.6.1", "aidev-evolve/0.6.0 (space-check)"])
def test_an_event_from_the_frozen_evolve_client_or_newer_is_taken(client, ua):
    assert client.post("/api/evolve/submit", json=seed(), headers=H("alice", ua)).status_code == 200


def test_the_minimum_evolve_version_is_the_part2_package_version():
    """One frozen version: the board's minimum is what the Part 2 package (and so its README command) ships."""
    init = (paths.evolve_dir() / "src" / "aidev_evolve" / "__init__.py").read_text()
    version = tuple(int(x) for x in re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', init).groups())
    assert app_mod.MIN_EVOLVE == version


def test_the_part2_client_names_its_version_on_every_event():
    board = (paths.evolve_dir() / "src" / "aidev_evolve" / "board.py").read_text()
    assert 'f"aidev-evolve/{__version__}"' in board


def test_the_minimum_runner_is_the_part1_package_version():
    init = (paths.runner_dir() / "src" / "aidev_swarm" / "__init__.py").read_text()
    version = tuple(int(x) for x in re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', init).groups())
    assert app_mod.MIN_RUNNER == version


# ---- load-test names (staging) ---------------------------------------------------------------------------------------


def test_an_allowed_account_claims_and_submits_under_a_load_test_name(client, store):
    r = client.post("/api/claim", headers={**H("joel"), AS: "lt-07"})
    assert r.status_code == 200
    cid = r.json()["cell_id"]
    assert [(le["cell_id"], le["user"]) for le in store.snapshot()["leases"]] == [(cid, "lt-07")]
    assert client.post(f"/api/heartbeat/{cid}", headers={**H("joel"), AS: "lt-07"}).status_code == 200
    assert client.post(f"/api/heartbeat/{cid}", headers=H("joel")).status_code == 409  # joel himself does not hold it


def test_each_load_test_name_has_its_own_share_like_an_attendee(client):
    cells = {client.post("/api/claim", headers={**H("joel"), AS: f"lt-{i:02d}"}).json()["cell_id"] for i in range(4)}
    assert len(cells) == 4  # one account alone could not hold four fresh leases: its share starts at 2


def test_evolve_events_post_under_the_load_test_name_with_its_own_rate_limit(client):
    for i in range(12):  # more than one user's burst of 10
        r = client.post("/api/evolve/submit", json={**seed(), "run_id": f"lt-{i:02d}-run"}, headers={**H("joel", EVOLVE_UA), AS: f"lt-{i:02d}"})
        assert r.status_code == 200, r.text
    assert {r["user"] for r in client.get("/api/evolve/state").json()["runs"]} == {f"lt-{i:02d}" for i in range(12)}


@pytest.mark.parametrize("name", ["alice", "lt-", "lt-UPPER", "lt-" + "x" * 40, "lt-a b", ""])
def test_a_load_test_name_must_look_like_one(client, name):
    """Never a real username: the names live in their own `lt-` space."""
    assert client.post("/api/claim", headers={**H("joel"), AS: name}).status_code == 422


def test_other_accounts_cannot_use_a_load_test_name(client, store):
    r = client.post("/api/claim", headers={**H("mallory"), AS: "lt-07"})
    assert r.status_code == 403 and store.snapshot()["leases"] == []


def test_without_the_setting_nobody_can(store):
    c = TestClient(create_app(store, TokenAuth(), reference=[], meta={}))
    assert c.post("/api/claim", headers={**H("joel"), AS: "lt-07"}).status_code == 403
    assert c.post("/api/claim", headers=H("joel")).status_code == 200  # and nothing else changes


def _files(tmp_path):
    import json

    from conftest import GRID, PRICES
    (tmp_path / "grid.json").write_text(json.dumps(GRID))
    (tmp_path / "prices.json").write_text(json.dumps(PRICES))
    return ["--grid", str(tmp_path / "grid.json"), "--prices", str(tmp_path / "prices.json"), "--reference", str(tmp_path / "none.json")]


def test_the_coordinator_reads_the_load_test_accounts_from_its_environment(tmp_path):
    env = {"SPACE_ID": deploy.TARGETS["staging"][0], "SWARM_LOADTEST_USERS": "Joel, bob"}
    c = cli.build(cli.parse(_files(tmp_path)), env=env)
    assert c.loadtest_users == frozenset({"joel", "bob"})  # usernames compare lowercased


def test_the_workshop_space_ignores_the_setting(tmp_path, caplog):
    env = {"SPACE_ID": deploy.DEFAULT_SPACE, "SWARM_LOADTEST_USERS": "joel"}
    c = cli.build(cli.parse(_files(tmp_path)), env=env)
    assert c.loadtest_users == frozenset()
    assert "SWARM_LOADTEST_USERS" in caplog.text  # said loudly in the Space log


class VarHub(FakeHub):
    def __init__(self, variables=None):
        super().__init__()
        self.variables.update(variables or {})


def test_deploy_to_staging_can_allow_load_test_accounts(tmp_path):
    hub = VarHub()
    deploy.deploy(*deploy.TARGETS["staging"], api=hub, token_file=token_file(tmp_path), staging=True, loadtest_users="joel")
    assert hub.variables["SWARM_LOADTEST_USERS"] == "joel"


def test_a_plain_deploy_removes_the_setting_again(tmp_path):
    hub = VarHub({"SWARM_LOADTEST_USERS": "joel"})
    deploy.deploy(*deploy.TARGETS["staging"], api=hub, token_file=token_file(tmp_path), staging=True)
    assert "SWARM_LOADTEST_USERS" not in hub.variables
    hub = VarHub()  # nothing to remove: no call, no error
    deploy.deploy(*deploy.TARGETS["staging"], api=hub, token_file=token_file(tmp_path), staging=True)
    assert not any(c[0] == "delete_variable" for c in hub.calls)


def test_deploy_refuses_load_test_accounts_on_the_workshop_space_before_any_hub_call(tmp_path):
    hub = VarHub()
    with pytest.raises(SystemExit, match="staging"):
        deploy.deploy(*deploy.TARGETS["prod"], api=hub, token_file=token_file(tmp_path), loadtest_users="joel")
    assert hub.calls == [] and hub.secrets == {}


def test_the_board_tells_clients_the_minimum_versions_it_accepts(client):
    """What the Part 2 client asks at the start of a run, so an old command is noticed at once (GET, no auth)."""
    r = client.get("/api/versions")
    assert r.status_code == 200
    assert r.json() == {"aidev-swarm": ".".join(map(str, app_mod.MIN_RUNNER)), "aidev-evolve": ".".join(map(str, app_mod.MIN_EVOLVE))}


# ---- sized for the room: coupon links on the start pages, more trials per cell ----------------------------------------


def _pages(store, **kw):
    c = TestClient(create_app(store, TokenAuth(), reference=[], meta={}, **kw))
    return c.get("/join").text, c.get("/evolve/join").text


def test_private_links_never_reach_start_pages(tmp_path):
    secret = "https://huggingface.co/private-sentinel"
    c = cli.build(cli.parse(_files(tmp_path)), env={"SWARM_COUPON_PART1": secret, "SWARM_COUPON_PART2": secret, "SWARM_TRIALS_PER_CELL": "5"})
    client = TestClient(c.app)
    for path in ("/join", "/evolve/join"):
        assert secret not in client.get(path).text
        assert "Credits for today" not in client.get(path).text
    state = client.get("/api/state").json()
    assert state["counts"]["cells_total"] == 30 and state["grid"]["trials"] == 5
    assert all(len(row) == 10 for row in state["grid"]["rows"].values())


def test_more_trials_per_cell_keep_the_order_every_task_once_before_any_task_twice(tmp_path):
    c = cli.build(cli.parse(_files(tmp_path)), env={"SWARM_TRIALS_PER_CELL": "4"})
    first = [x.cell_id for x in c.store.grid.cells[:8]]
    assert first == ["opencode__t1__0", "opencode__t2__0", "opencode__t1__1", "opencode__t2__1", "opencode__t1__2", "opencode__t2__2", "opencode__t1__3", "opencode__t2__3"]


@pytest.mark.parametrize("bad", ["0", "x", "1", "11", "-3"])
def test_trials_per_cell_can_only_grow_within_bounds(tmp_path, bad):
    """Fewer than grid.json's trials would drop reference cells; a typo like 50 would make the grid unfillable."""
    with pytest.raises(SystemExit, match="SWARM_TRIALS_PER_CELL"):
        cli.build(cli.parse(_files(tmp_path)), env={"SWARM_TRIALS_PER_CELL": bad})


def test_deploy_sets_and_clears_the_room_settings(tmp_path):
    hub = VarHub()
    with pytest.raises(ValueError):
        deploy.deploy(*deploy.TARGETS["staging"], api=hub, settings={"SWARM_COUPON_PART1": "private"})
    for value in ("5", ""):
        deploy.deploy(*deploy.TARGETS["staging"], api=hub, token_file=token_file(tmp_path), staging=True, settings={"SWARM_TRIALS_PER_CELL": value})
        assert hub.variables.get("SWARM_TRIALS_PER_CELL") == (value or None)
