from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import pytest

from conftest import GRID, PRICES, MemorySinkSpy, make_result
from swarm_coordinator.grid import Grid, cell_id
from swarm_coordinator.store import LEASE_S, AtLimit, Conflict, Invalid, NotFound, Store

LEASE = LEASE_S
RUNNER_CLI = Path(__file__).resolve().parents[2] / "runner" / "src" / "aidev_swarm" / "cli.py"


def runner_heartbeat_s() -> float:
    """The runner's heartbeat interval (runner/src/aidev_swarm/cli.py HEARTBEAT_S): the lease is sized from it."""
    m = re.search(r"^HEARTBEAT_S = (\d+(?:\.\d+)?)", RUNNER_CLI.read_text(), re.M)
    assert m, "runner cli.py lost its HEARTBEAT_S constant"
    return float(m.group(1))


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


def test_limit_2_allows_a_second_lease_one_cell_per_claim(store):
    assert ids(store.claim("alice", limit=2)) == ["opencode__t1__0"]
    assert ids(store.claim("alice", limit=2)) == ["opencode__t2__0"]
    with pytest.raises(AtLimit):
        store.claim("alice", limit=2)


def test_limit_outside_1_to_max_leases_is_invalid(store):
    from swarm_coordinator.store import MAX_LEASES

    for bad in (0, MAX_LEASES + 1):
        with pytest.raises(Invalid):
            store.claim("alice", limit=bad)


# ---------------------------------------------------------------- parallel runners (aidev-swarm --parallel)


def test_limit_lets_one_runner_hold_several_leases_one_claim_at_a_time(store, clock):
    """A Job running P trials at once claims one cell per free slot with limit=P (each slot its own heartbeats)."""
    from swarm_coordinator.store import RAMP_S

    store.claim("alice", limit=4)
    clock.t += 3 * RAMP_S  # past the slow start
    got = [store.claim("alice", limit=4) for _ in range(3)]
    assert [len(g) for g in got] == [1, 1, 1]
    with pytest.raises(AtLimit) as e:
        store.claim("alice", limit=4)
    assert len(e.value.held) == 4


def test_slow_start_adds_one_lease_per_ramp_step_from_the_first_claim(store, clock):
    """The first arrivals of a room must not take every cell before the others are seen."""
    from swarm_coordinator.store import RAMP_S

    assert store.claim("alice", limit=8)[0]["share"] == 2
    store.claim("alice", limit=8)
    with pytest.raises(AtLimit) as e:
        store.claim("alice", limit=8)
    assert e.value.share == 2
    clock.t += RAMP_S
    assert store.claim("alice", limit=8)[0]["share"] == 3


def test_share_caps_a_users_leases_so_a_full_room_does_not_overclaim(store, clock):
    """12 cells, 3 active attendees: nobody may hold more than ceil(12 / 3) = 4, whatever limit their runner asks."""
    from swarm_coordinator.store import RAMP_S

    store.claim("alice", limit=8)
    store.claim("bob")
    store.claim("carol")
    clock.t += 10 * RAMP_S
    store.heartbeat("opencode__t1__0", "alice")
    store.heartbeat("opencode__t2__0", "bob")
    store.heartbeat("opencode__t1__1", "carol")
    got = 1
    with pytest.raises(AtLimit) as e:
        for _ in range(8):
            got += len(store.claim("alice", limit=8))
    assert got == 4 and e.value.share == 4


def test_an_attendee_who_went_quiet_stops_counting_for_the_share(store, clock):
    from swarm_coordinator.store import ACTIVE_S

    c = store.claim("bob")[0]
    store.release(c["cell_id"], "bob")
    first = store.claim("alice", limit=8)[0]
    clock.t += 1000  # alice is past her slow start: only the room limits her share
    store.heartbeat(first["cell_id"], "alice")
    store.claim("bob")  # bob is back: 2 active
    assert store.claim("alice", limit=8)[0]["share"] == 6
    clock.t += ACTIVE_S + 1  # bob neither claims nor heartbeats, and his lease has run out
    store.heartbeat(first["cell_id"], "alice")
    assert store.claim("alice", limit=8)[0]["share"] == 12


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


# ---------------------------------------------------------------- short leases (Joel, 18 Nov: a dead job's cell returns at once)


def test_a_dead_runners_cell_returns_to_the_queue_within_two_minutes(store, clock):
    c = store.claim("alice")[0]
    clock.advance(30)
    store.heartbeat(c["cell_id"], "alice")  # its last beat, then the Job is killed (no release)
    clock.advance(LEASE_S + 1)
    assert LEASE_S + 1 <= 120
    assert ids(store.claim("bob")) == [c["cell_id"]]


def test_the_lease_outlasts_two_missed_runner_heartbeats():
    beat = runner_heartbeat_s()
    assert 10 <= beat and 2 * beat < LEASE_S <= 120


def test_a_slow_trial_that_keeps_heartbeating_never_loses_its_lease(store, clock):
    """A 20-minute trial beats every HEARTBEAT_S (one beat lost now and then); nobody else ever gets its cell."""
    beat = runner_heartbeat_s()
    c = store.claim("alice")[0]
    t = 0.0
    for k in range(int(20 * 60 / beat)):
        clock.advance(beat)
        t += beat
        if k % 7 != 3:  # every seventh beat is lost on the way
            store.heartbeat(c["cell_id"], "alice")
        got = store.claim(f"bob{k}")
        assert c["cell_id"] not in ids(got), f"alice lost her lease after {t:.0f} s"
        for g in got:
            store.release(g["cell_id"], f"bob{k}")
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    assert [r["cell_id"] for r in store.results()] == [c["cell_id"]]


def test_a_state_read_shows_an_expired_lease_as_free(store, clock):
    c = store.claim("alice")[0]
    assert [le["cell_id"] for le in store.snapshot()["leases"]] == [c["cell_id"]]
    clock.advance(LEASE_S + 1)
    assert store.snapshot()["leases"] == []


def test_leased_open_counts_cells_someone_is_still_working_on(store, clock):
    a = store.claim("alice")[0]
    store.claim("bob")
    assert store.leased_open() == 2
    store.submit(a["cell_id"], "alice", make_result(a["cell_id"]))
    assert store.leased_open() == 1
    clock.advance(LEASE_S + 1)
    assert store.leased_open() == 0


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


def test_late_result_after_the_cell_was_reclaimed_is_rejected_and_logged(store, clock, caplog):
    c = store.claim("alice")[0]
    clock.advance(LEASE + 1)
    assert ids(store.claim("bob")) == [c["cell_id"]]
    with caplog.at_level(logging.WARNING, logger="swarm.store"), pytest.raises(Conflict):
        store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    assert store.results() == []
    assert any("late result" in r.getMessage() and "alice" in r.getMessage() and c["cell_id"] in r.getMessage() and "bob" in r.getMessage()
               for r in caplog.records)
    store.submit(c["cell_id"], "bob", make_result(c["cell_id"], user="bob"))
    assert [r["hf_user"] for r in store.results()] == ["bob"]


def test_late_result_is_accepted_when_the_reclaimer_gave_the_cell_back(store, clock):
    c = store.claim("alice")[0]
    clock.advance(LEASE + 1)
    store.claim("bob")
    store.release(c["cell_id"], "bob")
    store.submit(c["cell_id"], "alice", make_result(c["cell_id"]))
    assert [r["hf_user"] for r in store.results()] == ["alice"]


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


def test_restored_leases_get_a_fresh_lease_so_live_runners_survive_a_slow_restart(grid, clock):
    """A Space restart takes longer than a lease: a saved lease restarts its clock at the restore, so a live
    runner's next heartbeat keeps its cell; a runner that died meanwhile loses it one lease later."""
    sink = MemorySinkSpy()
    s1 = Store(grid, PRICES, clock=clock, sink=sink)
    a, b = s1.claim("alice")[0], s1.claim("bob")[0]
    clock.advance(5 * 60)  # the restart
    s2 = Store(grid, PRICES, clock=clock, sink=sink)
    s2.restore()
    assert a["cell_id"] not in ids(s2.claim("carol"))
    clock.advance(LEASE_S - 5)
    s2.heartbeat(a["cell_id"], "alice")
    clock.advance(10)  # bob never beat again
    assert ids(s2.claim("dave")) == [b["cell_id"]]


def test_restore_keeps_results_of_cells_outside_the_grid_out_of_the_queue(grid, clock):
    sink = MemorySinkSpy(results=[make_result("gone__t9__0")])
    s = Store(grid, PRICES, clock=clock, sink=sink)
    s.restore()
    assert ids(s.claim("alice")) == ["opencode__t1__0"]


def test_cell_id_format():
    assert cell_id("mini-swe-agent", "kv-store-grpc", 2) == "mini-swe-agent__kv-store-grpc__2"
