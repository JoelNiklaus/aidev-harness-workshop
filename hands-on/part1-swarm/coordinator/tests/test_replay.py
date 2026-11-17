from __future__ import annotations

import json
from pathlib import Path

from conftest import PRICES, MemorySinkSpy, make_result
from swarm_coordinator.grid import Grid
from swarm_coordinator.paths import grid_path, reference_path
from swarm_coordinator.replay import Replayer, load_results
from swarm_coordinator.store import Store, validate


def results_for(grid, wall=600.0):
    out = []
    for c in grid.cells:
        out.append(make_result(c.cell_id, user="harness-leaderboard", passed=c.trial == 0, wall_s=wall))
    return out


def test_schedule_follows_claim_order_and_fits_the_duration(grid, clock):
    store = Store(grid, PRICES, clock=clock, sink=MemorySinkSpy())
    rp = Replayer(store, results_for(grid), minutes=10, concurrency=4, clock=clock)
    claims = [e for e in rp.events if e.kind == "claim"]
    assert [e.cell_id for e in claims] == [c.cell_id for c in grid.cells]
    assert max(e.at for e in rp.events) == 600.0  # the last result lands at the end of the 10 minutes
    assert {e.user for e in claims} == {f"replay-{i:03d}" for i in range(4)}


def test_stepping_moves_cells_from_claimed_to_done(grid, clock):
    store = Store(grid, PRICES, clock=clock, sink=MemorySinkSpy())
    rp = Replayer(store, results_for(grid), minutes=10, concurrency=4, clock=clock)
    rp.step()  # t=0: the first wave of 4 claims
    snap = store.snapshot()
    assert len(snap["leases"]) == 4 and snap["results"] == []
    clock.advance(600)
    rp.step()
    snap = store.snapshot()
    assert len(snap["results"]) == 12 and snap["leases"] == []
    assert rp.done
    assert {r["hf_user"] for r in snap["results"]} <= {f"replay-{i:03d}" for i in range(4)}


def test_cells_without_a_recorded_result_are_skipped(grid, clock):
    store = Store(grid, PRICES, clock=clock, sink=MemorySinkSpy())
    rp = Replayer(store, results_for(grid)[:5], minutes=1, concurrency=2, clock=clock)
    clock.advance(60)
    rp.step()
    assert len(store.results()) == 5


def test_replay_needs_no_network(grid, clock, monkeypatch):
    import socket

    def no_net(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.setattr(socket.socket, "connect", no_net)
    store = Store(grid, PRICES, clock=clock, sink=None)
    rp = Replayer(store, results_for(grid), minutes=1, concurrency=12, clock=clock)
    clock.advance(60)
    rp.step()
    assert len(store.results()) == 12


def test_shipped_reference_file_holds_valid_results():
    results = load_results(reference_path())
    assert len(results) >= 100
    for r in results:
        validate(r)
        h, t, k = r["cell_id"].split("__")
        assert (h, t, int(k)) == (r["harness"], r["task"], r["trial"])
    doc = json.loads(Path(reference_path()).read_text())
    assert doc["model"] == "zai-org/GLM-5.3-Flash" and doc["effort"] == "low"


def test_shipped_reference_covers_every_grid_cell_and_nothing_else():
    # the board's "Ref" column and hollow markers must use the same tasks as the room's grid;
    # a task missing from the reference silently skews the comparison
    grid = Grid.load(grid_path())
    results = load_results(reference_path())
    cells = {(r["harness"], r["task"]) for r in results}
    missing = {(h, t) for h in grid.harnesses for t in grid.tasks} - cells
    assert not missing, f"reference lacks {sorted(missing)[:5]}"
    assert {t for _, t in cells} <= set(grid.tasks)
