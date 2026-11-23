"""The workshop run (0.6.0, the default of `aidev-evolve run`): the seed's scores ship in the wheel, one iteration,
and one HF Job that measures the patch on validation (the re-measure alongside) and seed and patch on held-out.

Why: with 40 loops at once the four Jobs of the 0.5.3 run (seed, iteration, re-measure, held-out) took 29 min at the
median (PROGRESS.md, 21 Nov); the session is 40 min. `run --full` keeps the long version (tests/test_loop.py).
"""

import hashlib
import json
from pathlib import Path

import pytest

from aidev_evolve import calibration, cli, launch, loop, stats, strategy
from test_loop import SPLITS, FakeBoard, FakeProposer, add_fix, outcome


def record(cand, task, i, result, traj=True):
    r = {"candidate": cand, "task": task, "trial_name": f"{task}__{i}", "outcome": result, "reward": 0.0, "wall_s": 60.0, "cost_usd": 0.003}
    if traj:
        r["trajectory"] = {"task": task, "end": "returned", "error": None, "steps": [], "usage": {"calls": 1}, "agent_s": 30, "budget_s": 150}
    return r


def shipped(seed_source=loop.SEED_SOURCE, sets=2, val_trials=4):
    """A calibration as the wheel ships it, for the fake benchmark of test_loop (the seed passes s2, v4 and h2)."""
    recs = [record("c000", t, i, outcome(seed_source, t, i)) for t in SPLITS["search"] for i in range(sets)]
    recs += [record("c000", t, i, outcome(seed_source, t, i)) for t in SPLITS["validation"] for i in range(val_trials)]
    return calibration.build(recs, SPLITS, seed_source=seed_source, budget_s=150, sets=sets, measured={"jobs": ["job0"], "version": "0"})


class Eval:
    """The fake benchmark; `rule(source, task, i)` overrides one outcome."""

    def __init__(self, rule=None):
        self.specs, self.rule = [], rule

    def __call__(self, spec, on_record):
        self.specs.append(spec)
        recs = []
        for entry in spec["runs"]:
            src = spec["candidates"][entry["candidate"]]
            for task in entry["tasks"]:
                for i in range(entry["trials"]):
                    res = (self.rule and self.rule(src, task, i)) or outcome(src, task, i)
                    recs.append(record(entry["candidate"], task, i, res))
                    on_record(recs[-1])
        return launch.EvalResult(recs, f"job{len(self.specs)}", None, 120.0, 0.0004)


def make(tmp_path, edits, *, rule=None, preset="baseline", run_id=None, resume=False, **knobs):
    knobs = {"val_trials": 1, "z": 1.0, **knobs}
    ev, pr, board, out = Eval(rule), FakeProposer(edits), FakeBoard(), []
    evo = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=ev, propose=pr, board=board, write=out.append, budget_s=150, run_id=run_id,
                       strategy=strategy.with_knobs(strategy.load(preset), **knobs), calibration=shipped(), fast=True, resume=resume)
    return evo, ev, pr, board, out


def runs_of(spec):
    return {(r["candidate"], tuple(r["tasks"])): r["trials"] for r in spec["runs"]}


# ---- the shipped seed calibration ---------------------------------------------------------------------------------------

def test_the_shipped_seed_scores_replace_the_seed_job(tmp_path):
    evo, ev, pr, board, out = make(tmp_path, [])
    evo.run(iterations=0)
    assert ev.specs == []  # nothing measured, nothing spent
    assert evo.state["spend_usd"] == 0
    assert evo.state["margin"] == pytest.approx(0.25) and evo.state["flip_rate"] == 0  # 1 of 4 tasks; the fake seed never flips
    assert evo.val_score("c000") == pytest.approx(0.25) and evo.search_score("c000") == pytest.approx(0.5)
    text = " ".join("".join(out).split())
    assert "Seed agent: search 50%, validation 25% (measured before the workshop, 4 trials per validation task" in text
    assert [e["kind"] for e in board.events] == ["seed"] and board.events[0]["val_score"] == pytest.approx(0.25)


def test_the_proposer_reads_one_shipped_set_of_seed_trajectories(tmp_path):
    evo, ev, pr, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1, heldout_trials=1)
    traces = [f for f in pr.seen[0]["files"] if f.startswith("traces/") and not f.endswith("INDEX.md")]
    assert len(traces) == len(SPLITS["search"])  # one trajectory per search task, not one per shipped set
    on_disk = [json.loads(l) for l in (evo.run_dir / "records.jsonl").read_text().splitlines()]
    seed_search = [r for r in on_disk if r["eval"] == "e000" and r["split"] == "search"]
    assert len(seed_search) == len(SPLITS["search"]) and all(r["trajectory"] for r in seed_search)


def test_runs_pick_different_shipped_sets_by_their_id_and_always_the_same_one():
    cal = shipped(sets=4)
    picks = {calibration.pick(cal, f"alice-run{i}") for i in range(40)}
    assert picks == {0, 1, 2, 3}
    assert calibration.pick(cal, "alice-run7") == calibration.pick(cal, "alice-run7")


def test_a_calibration_of_another_seed_or_budget_is_refused(tmp_path):
    with pytest.raises(ValueError, match="another seed"):
        loop.Evolver(tmp_path / "a", splits=SPLITS, evaluate=Eval(), propose=FakeProposer([]), budget_s=150, calibration=shipped(), seed_source=loop.SEED_SOURCE + "# x\n")
    with pytest.raises(ValueError, match="budget"):
        loop.Evolver(tmp_path / "b", splits=SPLITS, evaluate=Eval(), propose=FakeProposer([]), budget_s=240, calibration=shipped())


def test_the_bundled_calibration_fits_the_bundled_seed_and_splits():
    cal, splits = calibration.bundled(), loop.default_splits()
    assert cal["seed_sha256"] == hashlib.sha256(loop.SEED_SOURCE.encode()).hexdigest()
    assert cal["budget_s"] == splits["budget_s"]
    assert len(cal["search_sets"]) >= 4
    for s in cal["search_sets"]:
        assert sorted(r["task"] for r in s) == sorted(splits["search"])
        assert all(r["candidate"] == "c000" and r["trajectory"]["steps"] for r in s)
    pt = stats.per_task(cal["validation"], "c000")
    assert sorted(pt) == sorted(splits["validation"]) and min(len(v) for v in pt.values()) >= 6
    assert not any("trajectory" in r for r in cal["validation"])  # validation trajectories never reach the proposer's machine
    assert cal["measured"]["jobs"] and cal["measured"]["version"]


def test_flip_rate_counts_every_pair_of_trials():
    assert stats.flip_rate({"a": [1, 0], "b": [1, 1]}) == pytest.approx(0.5)  # as before with two trials
    assert stats.flip_rate({"a": [1, 1, 0, 0]}) == pytest.approx(4 / 6)  # 4 of the 6 pairs disagree
    assert stats.flip_rate({"a": [1]}) == 0


# ---- one Job ----------------------------------------------------------------------------------------------------------------

def test_a_workshop_run_is_one_job_with_validation_and_heldout_together(tmp_path):
    evo, ev, pr, board, out = make(tmp_path, [("Fix it", add_fix)], val_trials=2, confirm=True)
    evo.run(iterations=1, heldout_trials=1)
    assert len(ev.specs) == 1
    val, ho = tuple(SPLITS["validation"]), tuple(SPLITS["heldout"])
    assert runs_of(ev.specs[0]) == {("c001", val): 4,  # the re-measure runs alongside: twice the validation trials
                                    ("c000", val): 2,  # the parent again in the same Job (drift), pooled with the shipped trials
                                    ("c000", ho): 1, ("c001", ho): 1}  # held-out for seed and patch before the verdict is known
    h = evo.state["history"][0]
    assert h["verdict"] == "kept" and evo.state["incumbent"] == "c001"
    ho_res = evo.state["heldout"]
    assert ho_res["best_candidate"] == "c001" and ho_res["trials"] == 1 and ho_res["diff"] == pytest.approx(0.5)
    assert [e["kind"] for e in board.events] == ["seed", "iteration", "heldout"]
    text = "".join(out)
    assert text.index("KEPT") < text.index("Held-out check")
    assert (evo.run_dir / "final_agent.py").read_text().endswith("# FIX\n")


def test_the_last_iteration_does_not_measure_search_tasks_nobody_will_read(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1, heldout_trials=1)
    assert not any(set(r["tasks"]) & set(SPLITS["search"]) for r in ev.specs[0]["runs"])
    assert evo.state["history"][0]["search_child"] is None


def test_a_dropped_patch_ends_with_the_seed_and_its_heldout_trials_are_not_used(tmp_path):
    evo, ev, pr, board, out = make(tmp_path, [("Break it", lambda s: s + "\n# BREAK\n")])
    evo.run(iterations=1, heldout_trials=1)
    assert len(ev.specs) == 1
    assert evo.state["history"][0]["verdict"] == "dropped" and evo.state["incumbent"] == "c000"
    ho = evo.state["heldout"]
    assert ho["best_candidate"] == "c000" and ho["diff"] == 0 and ho["seed"] == ho["best"]
    assert all(v[2:] == v[:2] for v in ho["per_task"].values())  # the final agent's counts are the seed's, not the dropped patch's
    assert (evo.run_dir / "final_agent.py").read_text() == loop.SEED_SOURCE


def test_the_verdict_uses_every_validation_trial_of_the_job(tmp_path):
    """A patch that passes v1 in its first trial only: +25 points on one trial (the margin), +12.5 pooled over two."""
    lucky = lambda src, task, i: "pass" if "# LUCKY" in src and task == "v1" and i == 0 else None  # noqa: E731
    one, *_ = make(tmp_path / "one", [("Lucky", lambda s: s + "\n# LUCKY\n")], rule=lucky, confirm=False)
    one.run(iterations=1, heldout_trials=1)
    assert one.state["history"][0]["verdict"] == "kept"  # one trial: 25% -> 50%
    two, ev, *_ = make(tmp_path / "two", [("Lucky", lambda s: s + "\n# LUCKY\n")], rule=lucky, confirm=True)
    two.run(iterations=1, heldout_trials=1)
    h = two.state["history"][0]
    assert h["val_child"] == pytest.approx(0.375) and h["verdict"] == "dropped"  # pooled: 25% -> 37.5%, under the margin of one task
    assert len(ev.specs) == 1


def test_a_skipped_proposal_still_ends_with_a_heldout_result(tmp_path):
    evo, ev, pr, board, _ = make(tmp_path, [("Nothing", None)])
    evo.run(iterations=1, heldout_trials=1)
    assert evo.state["history"][0]["verdict"].startswith("skipped")
    assert len(ev.specs) == 1 and runs_of(ev.specs[0]) == {("c000", tuple(SPLITS["heldout"])): 1}
    assert evo.state["heldout"]["best_candidate"] == "c000"
    assert [e["kind"] for e in board.events] == ["seed", "iteration", "heldout"]


def test_two_candidates_keep_their_separate_remeasure(tmp_path):
    """Picking the better of two is a selection: its winner is measured again in a Job of its own, as in --full."""
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix), ("Other", lambda s: s + "\n# other\n")], preset="two-candidates")
    evo.run(iterations=1, heldout_trials=1)
    assert len(ev.specs) == 3  # screen, re-measure, held-out
    assert evo.state["incumbent"] == "c001a" and evo.state["heldout"]["best_candidate"] == "c001a"


def test_a_resumed_workshop_run_does_not_load_the_seed_twice(tmp_path):
    evo, ev, *_ = make(tmp_path, [])
    evo.run(iterations=0)
    n = len((evo.run_dir / "records.jsonl").read_text().splitlines())
    again, ev2, *_ = make(tmp_path, [("Fix it", add_fix)], resume=True)
    again.run(iterations=1, heldout_trials=1)
    assert again.fast and len(ev2.specs) == 1
    seed_recs = [r for r in again.records if r["eval"] == "e000"]
    assert len(seed_recs) == n


# ---- the command line ---------------------------------------------------------------------------------------------------

@pytest.fixture
def evolver_args(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    seen = {}
    real = loop.Evolver

    def spy(run_dir, **kw):
        seen.update(kw)
        return real(run_dir, **kw)

    monkeypatch.setattr(loop, "Evolver", spy)

    def build(*flags):
        args = cli.build_parser().parse_args(["run", "--run-dir", str(tmp_path / f"r{len(flags)}{'-'.join(flags)[:20].replace('/', '_')}"), "--no-board", *flags])
        cli.make_evolver(args, Path(args.run_dir))
        return args, dict(seen)

    return build


def test_run_is_the_workshop_run_by_default(evolver_args):
    args, kw = evolver_args()
    assert kw["fast"] is True and kw["calibration"] == calibration.bundled() and kw["session_s"] is None
    assert cli.run_plan(args) == {"iterations": 1, "heldout_trials": 1}


def test_full_measures_the_seed_and_keeps_the_long_run(evolver_args):
    args, kw = evolver_args("--full")
    assert kw["fast"] is False and kw["calibration"] is None and kw["session_s"] == strategy.SESSION_MINUTES * 60
    assert cli.run_plan(args) == {"iterations": None, "heldout_trials": 2}


def test_measure_seed_keeps_the_one_job_run_but_measures_the_seed(evolver_args):
    args, kw = evolver_args("--measure-seed")
    assert kw["fast"] is True and kw["calibration"] is None


def test_an_own_seed_or_own_splits_are_measured_because_the_shipped_scores_are_not_theirs(evolver_args, tmp_path):
    seed = tmp_path / "mine.py"
    seed.write_text(loop.SEED_SOURCE + "# mine\n")
    assert evolver_args("--seed", str(seed))[1]["calibration"] is None
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({**SPLITS, "budget_s": 150}))
    assert evolver_args("--splits", str(splits))[1]["calibration"] is None
    assert evolver_args("--budget-s", "240")[1]["calibration"] is None


def test_the_strategies_listing_says_what_the_workshop_run_costs_and_names_full(capsys):
    assert cli.main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "1 iteration" in out and "--full" in out and "1 held-out trial" in out


def test_the_workshop_estimate_is_one_job_and_well_inside_the_session():
    for name in strategy.BLOCKS.values():
        s = strategy.load(name)
        est, full = strategy.estimate(s), strategy.estimate(s, full=True)
        assert est["jobs"] == 1 and est["iterations_fit"] == 1
        assert est["minutes"] <= 18 and est["usd"] < full["usd"] and est["usd"] <= 0.75
        n_v, n_h = 8, 21
        assert est["trials"] == n_v * s.val_trials * (2 if s.confirm else 1) + n_v * s.val_trials + 2 * n_h * strategy.WORKSHOP_HELDOUT_TRIALS


def test_the_preset_hashes_did_not_change_with_the_workshop_run():
    """The board pools runs by hash and the recorded runs carry these: a comment may change, a knob or a prompt must not."""
    assert {n: strategy.load(n).hash for n in strategy.BLOCKS.values()} == {"baseline": "81639b8daa", "every-weakness": "290bff2951", "code-audit": "c7ffe234ad"}
