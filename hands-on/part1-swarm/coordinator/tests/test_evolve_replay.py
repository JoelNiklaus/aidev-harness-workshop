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


def test_shipped_recording_holds_the_phase_3_strategy_runs():
    # Phase 3, batch loops2 (talk folder evolve/runs/p3): 13 runs of 4 presets plus an edited copy, config C
    runs = load_recorded(paths.evolve_replay_path())
    assert len(runs) == 13
    assert {r["events"][0]["strategy"]["preset"] for r in runs} == {"baseline", "failures-first", "small-diffs", "two-candidates"}
    for r in runs:
        kinds = [e["kind"] for e in r["events"]]
        assert kinds[0] == "seed" and kinds[-1] == "heldout" and 1 <= kinds.count("iteration") <= 2
        ts = [e["t"] for e in r["events"]]
        assert ts == sorted(ts) and ts[0] > 0
        for e in r["events"]:
            v = evolve.validate({k: x for k, x in e.items() if k != "t"})
            assert v["strategy"]["hash"] == r["events"][0]["strategy"]["hash"]
        ho = evolve.validate({k: x for k, x in r["events"][-1].items() if k != "t"})
        assert len(ho["per_task"]) == 23 and ho["kept"]
        assert all(e.get("hypothesis") for e in r["events"] if e["kind"] == "iteration")


def test_replayed_strategy_runs_rank_like_the_phase_3_report():
    # p3-numbers.md, minimum 5 runs: no single strategy has 5 runs; pooled by preset, baseline (3 + 3 edited copies) has 6
    clock, store = Clock(), evolve.EvolveStore(clock=Clock())
    rp = EvolveReplayer(store, load_recorded(paths.evolve_replay_path()), minutes=1, clock=clock)
    clock.advance(61)
    rp.step()
    s = evolve.build_state(store.events(), [], mode="replay")
    assert s["strategies"]["n_runs"] == 13 and s["strategies"]["groups"] == []
    assert s["presets"]["groups"] == [["baseline"]] and s["presets"]["winner"]["patch"]["iterations"]


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
    assert st["leader"] == [[0, 0.42], [1, 0.42], [1, 0.86], [2, 0.86]]


def test_reference_curves_are_the_same_step_lines_as_live_runs():
    ref = reference_curves(recorded())
    assert ref == [{"label": "recorded B1", "curve": [[0, 0.42], [1, 0.42], [1, 0.86], [2, 0.86]]},
                   {"label": "recorded B2", "curve": [[0, 0.33], [1, 0.33], [1, 0.83]]}]


def test_replay_needs_a_positive_duration():
    with pytest.raises(ValueError):
        EvolveReplayer(evolve.EvolveStore(), recorded(), minutes=0)
