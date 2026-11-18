"""The loop driven by an optimization strategy: evidence, prompt, candidates per iteration, keep rule, time box,
and the strategy + held-out data the board needs to rank strategies."""

import json

import pytest

from aidev_evolve import loop, strategy

from test_loop import SPLITS, FakeBoard, FakeEval, FakeProposer, add_fix


def build(tmp_path, edits, strat=None, clock=None, session_s=None, **knobs):
    s = strat or strategy.load("baseline")
    s = strategy.with_knobs(s, **{"val_trials": 1, "z": 1.0, **knobs})
    ev, pr, board, out = FakeEval(), FakeProposer(edits), FakeBoard(), []
    kw = {"clock": clock} if clock else {}
    evo = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=ev, propose=pr, board=board, write=out.append, budget_s=150,
                       strategy=s, session_s=session_s, **kw)
    return evo, ev, pr, board, out


def test_the_strategy_prompt_goes_first_and_the_fixed_rules_are_always_appended(tmp_path):
    evo, ev, pr, *_ = build(tmp_path, [("Fix it", add_fix)], strat=strategy.load("failures-first"))
    evo.run(iterations=1)
    prompt = pr.seen[0]["prompt"]
    assert "FAILURES.md" in prompt  # the preset's approach
    assert "PATCH.md" in prompt and "never mention benchmark task names" in prompt and "No open/eval/exec" in prompt
    assert prompt.index("FAILURES.md") < prompt.index("PATCH.md")


def test_failed_only_evidence_hides_passing_transcripts_and_max_traces_caps_them(tmp_path):
    evo, ev, pr, *_ = build(tmp_path, [("Fix it", add_fix)], traces="failed")
    evo.run(iterations=1)
    files = pr.seen[0]["files"]
    assert "traces/s1__1.md" in files and "traces/s2__1.md" not in files  # the seed fails s1, passes s2
    evo2, _, pr2, *_ = build(tmp_path / "b", [("Fix it", add_fix)], traces="all", max_traces=1)
    evo2.run(iterations=1)
    assert [f for f in pr2.seen[0]["files"] if f.startswith("traces/") and f != "traces/INDEX.md"] == ["traces/s1__1.md"]


def test_history_modes_and_diffs(tmp_path):
    edits = [("Break it", lambda s: s + "\n# BREAK\n"), ("Again", add_fix)]
    seen = {}
    for mode, diffs in (("none", False), ("titles", False), ("full", True)):
        evo, _, pr, *_ = build(tmp_path / mode, list(edits), history=mode, diffs=diffs)
        evo.run(iterations=2)
        seen[mode] = pr.seen[1]["history"]
    assert "Break it" not in seen["none"]
    assert "Break it" in seen["titles"] and "Hypothesis" not in seen["titles"]
    assert "Hypothesis" in seen["full"] and "+# BREAK" in seen["full"]


def test_two_candidates_are_proposed_measured_together_and_the_better_one_goes_on(tmp_path):
    edits = [("Break it", lambda s: s + "\n# BREAK\n"), ("Fix it", add_fix)]
    evo, ev, pr, board, out = build(tmp_path, edits, strat=strategy.load("two-candidates"), confirm=False)
    evo.calibrate_seed()
    h = evo.iterate()
    assert len(pr.seen) == 2 and "SECOND most common" in pr.seen[1]["prompt"] and "SECOND most common" not in pr.seen[0]["prompt"]
    first_eval = ev.specs[1]["runs"]
    cands = {r["candidate"] for r in first_eval}
    assert cands == {"c001a", "c001b", "c000"}  # both children and the parent in one Job
    assert h["candidate"] == "c001b" and h["verdict"] == "kept"  # the fix wins the screen and the keep rule
    assert len(ev.specs) == 3  # a screened pick is always re-measured, even with confirm off
    assert h["siblings"] == [{"candidate": "c001a", "title": "Break it", "val": pytest.approx(0.0), "verdict": "dropped: lost the screen"}]
    assert "lost the screen" in evo._history_md()
    assert any("also proposed c001a: Break it" in o for o in out)  # the terminal shows the loser too
    it = [e for e in board.events if e["kind"] == "iteration"][0]
    assert it["candidate"] == "c001b" and it["siblings"][0]["candidate"] == "c001a"


def test_two_candidates_go_on_with_one_when_the_other_fails_the_audit(tmp_path):
    edits = [("Use os", lambda s: "import os\n" + s), ("Fix it", add_fix)]
    evo, ev, *_ = build(tmp_path, edits, strat=strategy.load("two-candidates"))
    evo.calibrate_seed()
    h = evo.iterate()
    assert h["candidate"] == "c001b" and h["verdict"] == "kept"
    assert h["siblings"][0]["verdict"] == "rejected: audit"


def test_the_session_time_box_stops_new_iterations_and_still_runs_the_heldout_check(tmp_path):
    now = {"t": 0.0}
    evo, ev, pr, board, out = build(tmp_path, [("Fix it", add_fix)] + [(f"Try {k}", lambda src, k=k: src + f"\n# try {k}\n") for k in range(5)], clock=lambda: now["t"], session_s=1200, confirm=False)
    inner = evo.evaluate

    def slow_eval(spec, on_record):  # every Job takes 4 minutes of wall time
        now["t"] += 240
        return inner(spec, on_record)

    evo.evaluate = slow_eval
    st = evo.run(iterations=6, heldout_trials=1)
    # calibration ends at 4 min, iteration 1 (one Job) at 8: 8 + 4 + 5.5 held-out fits 20 min; after iteration 2 (12 min)
    # 12 + 4 + 5.5 does not: two iterations, then the held-out check
    assert len(st["history"]) == 2
    assert st["heldout"] is not None
    assert any("session time box" in o for o in out)


def test_a_remeasured_iteration_does_not_make_the_time_box_expect_every_iteration_to_be_that_long(tmp_path):
    """Real run (18 Nov): iteration 1 kept + re-measured took 7.5 min; using it as the estimate stopped every run after
    one iteration. Only iterations without a re-measure (or the prior) estimate the next one."""
    now = {"t": 0.0}
    edits = [("Fix it", add_fix)] + [(f"Try {k}", lambda src, k=k: src + f"\n# try {k}\n") for k in range(5)]
    evo, *_ = build(tmp_path, edits, clock=lambda: now["t"], session_s=22 * 60)
    inner = evo.evaluate

    def slow_eval(spec, on_record):
        now["t"] += 240
        return inner(spec, on_record)

    evo.evaluate = slow_eval
    st = evo.run(iterations=6, heldout_trials=1)
    # calibration 4 min, iteration 1 kept + re-measured ends at 12 min; 12 + 4.2 (prior) + 5.5 fits 22: a second iteration
    assert st["history"][0]["verdict"] == "kept" and len(st["history"]) == 2


def test_every_event_carries_the_strategy_and_the_heldout_event_its_per_task_counts(tmp_path):
    s = strategy.load("small-diffs")
    evo, ev, pr, board, out = build(tmp_path, [("Fix it", add_fix)], strat=s)
    evo.run(iterations=1, heldout_trials=2)
    for e in board.events:
        assert e["strategy"]["preset"] == "small-diffs" and e["strategy"]["knobs"]["val_trials"] == 1
    held = [e for e in board.events if e["kind"] == "heldout"][0]
    assert held["per_task"] == {"h1": [0, 2, 2, 2], "h2": [2, 2, 2, 2]}
    assert held["kept"] == ["Fix it"]
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())
    assert cfg["strategy"]["hash"] == evo.strategy.hash and cfg["strategy"]["prompt"] == s.prompt


def test_the_keep_rule_comes_from_the_strategy(tmp_path):
    evo, *_ = build(tmp_path, [], val_trials=2, z=2.0, confirm=False)
    assert (evo.val_trials, evo.z, evo.confirm) == (2, 2.0, False)


def test_a_resumed_run_keeps_its_strategy(tmp_path):
    evo, *_ = build(tmp_path, [("Fix it", add_fix)], strat=strategy.load("small-diffs"))
    evo.run(iterations=1)
    evo2 = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=FakeEval(), propose=FakeProposer([]), write=lambda s: None, resume=True)
    assert evo2.strategy.hash == evo.strategy.hash and evo2.z == evo.z
