"""Evolve replay: the recorded Phase 2 runs (B1, B2) stream into the /evolve board offline, and in live mode they
are the board's backdrop (reference curves)."""

from __future__ import annotations

import pytest

from swarm_coordinator import evolve, paths
from swarm_coordinator.evolve_replay import EvolveReplayer, load_recorded, reference_curves

from conftest import Clock
from test_evolve import heldout, it, seed


def recorded():
    return [
        {"name": "B1", "user": "recorded-B1", "events": [{"t": 100.0, **seed("B1", val=0.42)}, {"t": 400.0, **it(1, run_id="B1", accepted=True, best=0.88)},
                                                         {"t": 600.0, **it(2, run_id="B1", best=0.86)}, {"t": 1000.0, **heldout("B1")}]},
        {"name": "B2", "user": "recorded-B2", "events": [{"t": 50.0, **seed("B2", val=0.33)}, {"t": 2000.0, **it(1, run_id="B2", accepted=True, best=0.83)}]},
    ]


def test_shipped_recording_holds_two_complete_valid_runs():
    runs = load_recorded(paths.evolve_replay_path())
    assert [r["name"] for r in runs] == ["B1", "B2"]
    for r in runs:
        kinds = [e["kind"] for e in r["events"]]
        assert kinds == ["seed"] + ["iteration"] * 8 + ["heldout"]
        ts = [e["t"] for e in r["events"]]
        assert ts == sorted(ts) and ts[0] > 0
        for e in r["events"]:
            evolve.validate({k: v for k, v in e.items() if k != "t"})
        its = [e for e in r["events"] if e["kind"] == "iteration"]
        assert sum(e["accepted"] for e in its) == 1 and its[0]["accepted"]  # Phase 2: the step-limit fix, then drops
        assert all(e.get("hypothesis") for e in its)


def test_replay_schedule_is_squeezed_into_the_given_minutes():
    clock, store = Clock(), evolve.EvolveStore(clock=Clock())
    rp = EvolveReplayer(store, recorded(), minutes=10, clock=clock)
    assert rp.step() == 0
    clock.advance(10 * 60 * 400 / 2000 + 0.01)  # B1's first iteration, scaled: the longest run (2000 s) takes 10 min
    rp.step()
    assert [(e["user"], e["kind"]) for e in store.events()] == [("recorded-B2", "seed"), ("recorded-B1", "seed"), ("recorded-B1", "iteration")]
    clock.advance(10 * 60)
    rp.step()
    assert rp.done and len(store.events()) == 6


def test_replayed_events_build_the_board_like_live_ones():
    clock, store = Clock(), evolve.EvolveStore(clock=Clock())
    rp = EvolveReplayer(store, recorded(), minutes=1, clock=clock)
    clock.advance(61)
    rp.step()
    st = evolve.build_state(store.events(), reference=[], mode="replay")
    assert st["totals"]["attendees"] == 2 and st["totals"]["kept"] == 2
    assert st["room_best"] == [[0, 0.42], [1, 0.88], [2, 0.88]]


def test_reference_curves_are_the_recorded_seed_and_best_val_points():
    ref = reference_curves(recorded())
    assert ref == [{"label": "recorded B1", "curve": [[0, 0.42], [1, 0.88], [2, 0.86]]},
                   {"label": "recorded B2", "curve": [[0, 0.33], [1, 0.83]]}]


def test_replay_needs_a_positive_duration():
    with pytest.raises(ValueError):
        EvolveReplayer(evolve.EvolveStore(), recorded(), minutes=0)
