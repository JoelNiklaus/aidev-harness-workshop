"""Evolve replay: recorded runs (here B1, B2; shipped: the Phase 4 runs) stream into the /evolve board offline, and in live mode they
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


def test_shipped_recording_holds_the_phase_4_strategy_runs():
    # Phase 4 (talk folder evolve/runs/p4/loops): 20 runs, seed E / splits E (21 held-out tasks), presets
    # baseline (8), every-weakness (7), code-audit (5), all unedited
    runs = load_recorded(paths.evolve_replay_path())
    assert len(runs) == 20
    presets = [r["events"][0]["strategy"]["preset"] for r in runs]
    assert {p: presets.count(p) for p in set(presets)} == {"baseline": 8, "every-weakness": 7, "code-audit": 5}
    for r in runs:
        kinds = [e["kind"] for e in r["events"]]
        assert kinds[0] == "seed" and kinds[-1] == "heldout" and 1 <= kinds.count("iteration") <= 6
        ts = [e["t"] for e in r["events"]]
        assert ts == sorted(ts) and ts[0] > 0
        for e in r["events"]:
            v = evolve.validate({k: x for k, x in e.items() if k != "t"})
            assert v["strategy"]["hash"] == r["events"][0]["strategy"]["hash"] and not v["strategy"]["modified"]
        ho = evolve.validate({k: x for k, x in r["events"][-1].items() if k != "t"})
        assert len(ho["per_task"]) == 21 and ho["kept"]
        assert all(e.get("hypothesis") for e in r["events"] if e["kind"] == "iteration")


def test_replayed_strategy_runs_rank_like_the_phase_4_report():
    # part2-evolve 0.5.0 (commit 027f7be): code-audit wins alone on the 20 Phase 4 runs; every preset has 5+ runs
    clock, store = Clock(), evolve.EvolveStore(clock=Clock())
    rp = EvolveReplayer(store, load_recorded(paths.evolve_replay_path()), minutes=1, clock=clock)
    clock.advance(61)
    rp.step()
    s = evolve.build_state(store.events(), [], mode="replay")
    assert s["presets"]["n_runs"] == 20 and s["presets"]["groups"][0] == ["code-audit"]
    assert s["strategies"]["groups"][0] == ["c7ffe234ad"]  # by hash: the code-audit preset's hash
    w = s["presets"]["winner"]
    assert [x["name"] for x in w["strategies"]] == ["code-audit"] and w["patch"]["run_id"].startswith("code-audit-")
    assert w["patch"]["iterations"]


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


def _builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_evolve_replay", paths.root() / "replay" / "build_evolve_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builder_carries_the_provider_and_provider_calls_of_a_0_5_run(tmp_path):
    # a client-0.5.0 run folder: config.json names the attendee's providers, history and held-out count who answered
    import json
    d = tmp_path / "code-audit-9"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"providers": ["novita", "zai-org"], "strategy": {
        "name": "code-audit", "preset": "code-audit", "hash": "c7ffe234ad", "modified": False, "knobs": {}, "prompt": "Audit agent.py."}}))
    (d / "state.json").write_text(json.dumps({"curve": [0.2], "spend_usd": 1.5, "history": [
        {"iteration": 1, "candidate": "c001", "parent": "c000", "title": "t", "hypothesis": "h", "verdict": "kept", "val_child": 0.4,
         "val_parent": 0.2, "best_val": 0.4, "spend_usd": 0.5, "iter_usd": 0.3, "provider_calls": {"novita": 12, "zai-org": 1}}],
        "heldout": {"seed": 0.3, "best": 0.5, "best_candidate": "c001", "n_tasks": 21, "trials": 2, "diff": 0.2, "ci_low": 0.0,
                    "ci_high": 0.4, "per_task": {"t": [0, 2, 1, 2]}, "kept": ["t"], "provider_calls": {"novita": 40}}}))
    (d / "transcript.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"t": 10, "text": "Seed agent: search 20%"}, {"t": 50, "text": "Iteration 1 of 6"}, {"t": 90, "text": "done"})))
    events = _builder().run_events(d)
    assert {e["provider"] for e in events} == {"novita"}
    assert [e.get("provider_calls") for e in events] == [None, {"novita": 12, "zai-org": 1}, {"novita": 40}]
    for e in events:
        evolve.validate({k: x for k, x in e.items() if k != "t"})
