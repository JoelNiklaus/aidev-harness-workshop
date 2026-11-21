"""The real Coordinator client over real HTTP against the protocol stand-in (devtools/fake_coordinator.py),
plus a small simulated swarm: many clients claiming at once never get the same cell."""

import importlib.util
import threading
import time
from pathlib import Path

import pytest

from aidev_swarm.coordinator import Coordinator, LeaseLost

spec = importlib.util.spec_from_file_location("fake_coordinator", Path(__file__).parents[1] / "devtools" / "fake_coordinator.py")
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)

GRID = {"model": "m", "harnesses": ["pi", "opencode"], "tasks": ["t1", "t2", "t3"], "trials_per_cell": 2, "reasoning_effort": "low", "provider": "zai-org"}


@pytest.fixture
def server():
    srv, state = fake.serve(GRID)
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()


def test_order_is_one_harness_after_the_next_breadth_first_over_trials(server):
    url, _ = server
    c = Coordinator(url, "hf_token_a", retries=1)
    got = []
    while (cell := c.claim()) is not None:
        got.append(cell["cell_id"])
        c.submit(cell["cell_id"], {"cell_id": cell["cell_id"]})  # one live lease per user, as in the real coordinator
    assert got == ["pi__t1__0", "pi__t2__0", "pi__t3__0", "pi__t1__1", "pi__t2__1", "pi__t3__1",
                   "opencode__t1__0", "opencode__t2__0", "opencode__t3__0", "opencode__t1__1", "opencode__t2__1", "opencode__t3__1"]


def test_claim_heartbeat_submit_roundtrip_and_foreign_lease_refused(server):
    url, state = server
    a, b = Coordinator(url, "hf_token_a", retries=1), Coordinator(url, "hf_token_b", retries=1)
    cell = a.claim()
    assert cell["run_config"] == {"reasoning_effort": "low", "provider": "zai-org"}
    a.heartbeat(cell["cell_id"])
    with pytest.raises(LeaseLost):
        b.submit(cell["cell_id"], {"cell_id": cell["cell_id"]})
    a.submit(cell["cell_id"], {"cell_id": cell["cell_id"], "passed": True})
    assert state.state()["done"] == 1


def test_released_cell_is_handed_out_again_first(server):
    url, _ = server
    c = Coordinator(url, "hf_token_a", retries=1)
    first = c.claim()
    c.release(first["cell_id"])
    assert c.claim()["cell_id"] == first["cell_id"]


def test_expired_lease_returns_to_the_queue():
    srv, _ = fake.serve(GRID, lease_s=0.05)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        a, b = Coordinator(url, "hf_token_a", retries=1), Coordinator(url, "hf_token_b", retries=1)
        first = a.claim()
        time.sleep(0.1)
        assert b.claim()["cell_id"] == first["cell_id"]
    finally:
        srv.shutdown()


def test_simulated_swarm_of_30_clients_never_double_claims(server):
    url, _ = server
    claimed, lock = [], threading.Lock()

    def attendee(i):
        c = Coordinator(url, f"hf_token_{i:03d}", retries=3, backoff_s=0.01)
        while (cell := c.claim()) is not None:
            with lock:
                claimed.append(cell["cell_id"])
            c.submit(cell["cell_id"], {"cell_id": cell["cell_id"], "passed": i % 2 == 0})

    threads = [threading.Thread(target=attendee, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert sorted(claimed) == sorted(c["cell_id"] for c in fake.cells(GRID))


def test_retried_submit_by_the_same_user_is_idempotent_like_the_real_coordinator(server):
    url, state = server
    c = Coordinator(url, "hf_token_a", retries=1)
    cell = c.claim()
    c.submit(cell["cell_id"], {"cell_id": cell["cell_id"]})
    c.submit(cell["cell_id"], {"cell_id": cell["cell_id"]})  # a retry after a lost reply
    assert state.state()["done"] == 1
    with pytest.raises(LeaseLost):
        Coordinator(url, "hf_token_b", retries=1).submit(cell["cell_id"], {"cell_id": cell["cell_id"]})


def test_lease_limit_per_user_matches_the_real_coordinator(server):
    from aidev_swarm.coordinator import AtLimit

    url, _ = server
    c = Coordinator(url, "hf_token_a", retries=1)
    first = c.claim()
    with pytest.raises(AtLimit) as e:
        c.claim()
    assert e.value.held == [first["cell_id"]]
    assert c.claim(limit=2)["cell_id"] != first["cell_id"]  # limit=2 allows a second lease
    with pytest.raises(AtLimit) as e:
        c.claim(limit=2)
    assert e.value.share >= 2


def test_a_raised_lease_limit_lets_one_account_run_many_jobs():
    srv, _ = fake.serve(GRID, max_leases=25)
    try:
        c = Coordinator(f"http://127.0.0.1:{srv.server_address[1]}", "hf_token_a", retries=1)
        got = [c.claim()["cell_id"] for _ in range(5)]
        assert len(set(got)) == 5
    finally:
        srv.shutdown()


def test_state_counts_the_distinct_client_addresses(server):
    url, state = server
    Coordinator(url, "hf_token_a", retries=1).claim()
    assert state.state()["client_ips"] == 1


def test_the_stand_in_runs_outside_the_repo_as_in_an_hf_job():
    """The dry-run Job runs it as /tmp/fake_coordinator.py: a path with only two parents."""
    import subprocess
    import sys
    import uuid

    target = Path("/tmp") / f"fake_coordinator_{uuid.uuid4().hex[:8]}.py"
    target.write_text((Path(__file__).parents[1] / "devtools" / "fake_coordinator.py").read_text())
    try:
        p = subprocess.run([sys.executable, str(target), "--help"], capture_output=True, text=True, timeout=30)
    finally:
        target.unlink()
    assert p.returncode == 0, p.stderr


def test_behind_the_jobs_proxy_the_stand_in_accepts_requests_whose_token_the_proxy_consumed():
    import httpx

    srv, state = fake.serve(GRID, behind_jobs_proxy=True, max_leases=25)
    try:
        r = httpx.post(f"http://127.0.0.1:{srv.server_address[1]}/api/claim", headers={"X-Forwarded-For": "34.1.2.3, 10.0.0.1"})
        assert r.status_code == 200 and r.json()["cell_id"] == "pi__t1__0"
        assert state.state()["client_ips"] == 1 and state.clients == {"34.1.2.3"}
    finally:
        srv.shutdown()
