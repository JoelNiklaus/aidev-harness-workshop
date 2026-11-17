from __future__ import annotations

import threading

import pytest

from conftest import GRID, PRICES, MemorySinkSpy, make_result
from swarm_coordinator.grid import Grid, cell_id
from swarm_coordinator.store import AtLimit, Conflict, Invalid, NotFound, Store

LEASE = 40 * 60


def ids(claims):
    return [c["cell_id"] for c in claims]


# ---------------------------------------------------------------- grid order (Joel, 17 Nov)


def test_cells_are_handed_out_one_harness_after_the_next_breadth_first_over_trials(grid):
    order = [c.cell_id for c in grid.cells]
    assert order == [
        "opencode__t1__0", "opencode__t2__0", "opencode__t1__1", "opencode__t2__1",
        "openclaw__t1__0", "openclaw__t2__0", "openclaw__t1__1", "openclaw__t2__1",
        "pi__t1__0", "pi__t2__0", "pi__t1__1", "pi__t2__1",
    ]


def test_claims_follow_grid_order_across_users(store):
    got = [store.claim(f"user{i}")[0]["cell_id"] for i in range(6)]
    assert got == ["opencode__t1__0", "opencode__t2__0", "opencode__t1__1", "opencode__t2__1", "openclaw__t1__0", "openclaw__t2__0"]


def test_claim_carries_protocol_fields(store, clock):
    c = store.claim("alice")[0]
    assert c["harness"] == "opencode" and c["task"] == "t1" and c["trial"] == 0
    assert c["model"] == GRID["model"]
    assert c["lease_expires_at"].startswith("20")  # ISO timestamp
    assert c["run_config"]["benchmark"] == "terminal-bench-2.1"
    assert c["run_config"]["reasoning_effort"] == "low"
    assert c["run_config"]["provider"] == "zai-org"
    assert "_comment" not in c["run_config"] and "tasks" not in c["run_config"]


# ---------------------------------------------------------------- leases


def test_one_active_lease_per_user(store):
    first = store.claim("alice")
    with pytest.raises(AtLimit) as e:
        store.claim("alice")
    assert e.value.held == ids(first)


def test_n2_allows_two_active_leases(store):
    got = store.claim("alice", n=2)
    assert ids(got) == ["opencode__t1__0", "opencode__t2__0"]
    with pytest.raises(AtLimit):
        store.claim("alice", n=2)


def test_n2_tops_up_to_two(store):
    store.claim("alice")
    assert ids(store.claim("alice", n=2)) == ["opencode__t2__0"]


def test_n_above_two_is_invalid(store):
    with pytest.raises(Invalid):
        store.claim("alice", n=3)


def test_nothing_left_returns_empty(store):
    for i in range(12):
        store.claim(f"u{i}")
    assert store.claim("late") == []


def test_expired_lease_returns_to_the_queue_at_its_place(store, clock):
    store.claim("alice")  # opencode__t1__0
    store.claim("bob")    # opencode__t2__0
    clock.advance(LEASE + 1)
    assert ids(store.claim("carol")) == ["opencode__t1__0"]
    # alice's lease expired, so she may claim again
    assert ids(store.claim("alice")) == ["opencode__t2__0"]


def test_heartbeat_extends_the_lease(store, clock):
    c = store.claim("alice")[0]
    clock.advance(LEASE - 10)
    store.heartbeat(c["cell_id"], "alice")
    clock.advance(LEASE - 10)  # past the original expiry
    assert ids(store.claim("bob")) == ["opencode__t2__0"]


def test_heartbeat_by_another_user_conflicts(store):
    c = store.claim("alice")[0]
    with pytest.raises(Conflict):
        store.heartbeat(c["cell_id"], "bob")


def test_heartbeat_after_expiry_renews_when_nobody_took_the_cell(store, clock):
    c = store.claim("alice")[0]
    clock.advance(LEASE + 5)
    out = store.heartbeat(c["cell_id"], "alice")
    assert out["lease_expires_at"]
    assert ids(store.claim("bob")) == ["opencode__t2__0"]


def test_heartbeat_on_unknown_cell_is_not_found(store):
    with pytest.raises(NotFound):
        store.heartbeat("nope__t1__0", "alice")


def test_release_returns_the_cell(store):
    c = store.claim("alice")[0]
    store.release(c["cell_id"], "alice")
    assert ids(store.claim("bob")) == [c["cell_id"]]


def test_release_by_another_user_conflicts(store):
    c = store.claim("alice")[0]
    with pytest.raises(Conflict):
        store.release(c["cell_id"], "bob")


# ---------------------------------------------------------------- submit


def test_submit_by_lease_owner_records_the_result(store, sink):
    c = store.claim("alice")[0]
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"], user="mallory"))
    (r,) = store.results()
    assert r["hf_user"] == "alice"  # the authenticated user, not the body's claim
    assert sink.results[c["cell_id"]]["hf_user"] == "alice"
    # done cells are never handed out again; alice is free to claim
    assert ids(store.claim("alice")) == ["opencode__t2__0"]


def test_submit_recomputes_cost_from_prices(store):
    c = store.claim("alice")[0]
    out = store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    # 500k uncached x 0.15 + 500k cached x 0.03 + 10k out x 0.5, per 1M
    assert out["cost_usd"] == pytest.approx(0.075 + 0.015 + 0.005)
    assert out["cost_usd_reported"] == 99.0


def test_submit_while_another_user_holds_the_lease_conflicts(store):
    c = store.claim("alice")[0]
    with pytest.raises(Conflict):
        store.submit(c["cell_id"], "bob", make_result(c["cell_id"]))
    assert store.results() == []


def test_submit_must_match_the_cell(store):
    c = store.claim("alice")[0]
    with pytest.raises(Invalid):
        store.submit(c["cell_id"], "alice", make_result("opencode__t2__0"))
    with pytest.raises(Invalid):
        store.submit(c["cell_id"], "alice", {**make_result(c["cell_id"]), "task": "t2"})
    with pytest.raises(Invalid):
        store.submit(c["cell_id"], "alice", {**make_result(c["cell_id"]), "model": "other/model"})


def test_submit_missing_fields_is_invalid(store):
    c = store.claim("alice")[0]
    bad = make_result(c["cell_id"])
    del bad["passed"]
    with pytest.raises(Invalid):
        store.submit(c["cell_id"], "alice", bad)


def test_duplicate_submit_by_the_same_user_is_idempotent(store, sink):
    c = store.claim("alice")[0]
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    assert len(store.results()) == 1 and sink.put_calls == 1


def test_second_result_for_a_done_cell_by_another_user_conflicts(store, clock):
    c = store.claim("alice")[0]
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    with pytest.raises(Conflict):
        store.submit(c["cell_id"], "bob", make_result(c["cell_id"]))


def test_late_result_after_expiry_is_accepted_when_nobody_took_the_cell(store, clock):
    c = store.claim("alice")[0]
    clock.advance(LEASE + 60)
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    assert len(store.results()) == 1


def test_submit_on_unknown_cell_is_not_found(store):
    with pytest.raises(NotFound):
        store.submit("nope__t1__0", "alice", make_result("nope__t1__0"))


# ---------------------------------------------------------------- atomicity


def test_concurrent_claims_never_hand_out_a_cell_twice(grid, clock):
    big = Grid.from_dict({**GRID, "tasks": [f"t{i}" for i in range(40)], "trials_per_cell": 3})
    store = Store(big, PRICES, clock=clock, sink=MemorySinkSpy())
    got: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        for _ in range(10):
            try:
                c = store.claim(f"u{i}-{_}")
            except AtLimit:
                continue
            with lock:
                got.extend(ids(c))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == len(set(got)) == len(big.cells)


# ---------------------------------------------------------------- restart


def test_restart_rebuilds_from_the_sink(grid, clock):
    sink = MemorySinkSpy()
    s1 = Store(grid, PRICES, clock=clock, sink=sink)
    a = s1.claim("alice")[0]
    s1.submit(a["cell_id"], "alice", make_result(a["cell_id"]))
    b = s1.claim("bob")[0]
    s2 = Store(grid, PRICES, clock=clock, sink=sink)
    s2.restore()
    assert [r["cell_id"] for r in s2.results()] == [a["cell_id"]]
    # bob's lease survived the restart: his cell is not handed out again, and he can submit
    assert ids(s2.claim("carol")) == ["opencode__t1__1"]
    s2.submit(b["cell_id"], "bob", make_result(b["cell_id"]))
    assert len(s2.results()) == 2


def test_restore_keeps_results_of_cells_outside_the_grid_out_of_the_queue(grid, clock):
    sink = MemorySinkSpy(results=[make_result("gone__t9__0")])
    s = Store(grid, PRICES, clock=clock, sink=sink)
    s.restore()
    assert ids(s.claim("alice")) == ["opencode__t1__0"]


def test_cell_id_format():
    assert cell_id("mini-swe-agent", "kv-store-grpc", 2) == "mini-swe-agent__kv-store-grpc__2"
