"""The evolve loop end to end with fakes: seed calibration, propose, audit, preflight, paired evaluation,
keep or drop, traces for the next proposal, spend, the workshop board, the final held-out check."""

import json
from pathlib import Path

import pytest

from aidev_evolve import launch, loop, strategy
from aidev_evolve.proposer import ProposerResult

SPLITS = {"search": ["s1", "s2"], "validation": ["v1", "v2", "v3", "v4"], "heldout": ["h1", "h2"]}


def outcome(source, task, trial):
    """The fake benchmark: the seed passes v4 and s2; '# FIX' adds v1, v2, s1 and h1; '# BREAK' loses v4."""
    passes = {"v4", "s2", "h2"}
    if "# FIX" in source:
        passes |= {"v1", "v2", "s1", "h1"}
    if "# BREAK" in source:
        passes -= {"v4"}
    return "pass" if task in passes else "fail"


class FakeEval:
    def __init__(self):
        self.specs = []

    def __call__(self, spec, on_record):
        self.specs.append(spec)
        recs = []
        for entry in spec["runs"]:
            src = spec["candidates"][entry["candidate"]]
            for task in entry["tasks"]:
                for i in range(entry["trials"]):
                    r = {"candidate": entry["candidate"], "task": task, "trial_name": f"{task}__{i}", "outcome": outcome(src, task, i),
                         "reward": 0.0, "wall_s": 60.0, "cost_usd": 0.003,
                         "trajectory": {"task": task, "end": "returned", "error": None, "steps": [], "usage": {"calls": 1}, "agent_s": 30, "budget_s": 150}}
                    on_record(r)
                    recs.append(r)
        return launch.EvalResult(recs, f"job{len(self.specs)}", None, 120.0, 0.0004)


class FakeProposer:
    """Applies one scripted edit per iteration to work/agent.py and writes PATCH.md."""

    def __init__(self, edits):
        self.edits = list(edits)
        self.seen = []

    def __call__(self, work, prompt):
        work = Path(work)
        self.seen.append({"files": sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()), "prompt": prompt,
                          "history": (work / "HISTORY.md").read_text()})
        title, edit = self.edits.pop(0)
        if edit is not None:
            (work / "agent.py").write_text(edit((work / "agent.py").read_text()))
            (work / "PATCH.md").write_text(f"# {title}\n\nHypothesis: x\nChange: y\nExpected: z\n")
        return ProposerResult(True, 30.0, 0.2)


class FakeBoard:
    def __init__(self):
        self.events = []

    def submit(self, event):
        self.events.append(event)


def make(tmp_path, edits, **kw):
    knobs = {"val_trials": 1, "z": 1.0, **{k: kw.pop(k) for k in ("val_trials", "z", "confirm") if k in kw}}  # the smallest plan
    kw["strategy"] = strategy.with_knobs(strategy.load("baseline"), **knobs)  # the keep rule lives in the strategy
    ev, pr, board = FakeEval(), FakeProposer(edits), FakeBoard()
    out = []
    evo = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=ev, propose=pr, board=board, write=out.append, budget_s=150, **kw)
    return evo, ev, pr, board, out


def add_fix(src):
    return src + "\n# FIX\n"


def test_keeps_a_patch_that_wins_on_validation_and_drops_a_regression(tmp_path):
    evo, ev, pr, board, out = make(tmp_path, [("Fix it", add_fix), ("Break it", lambda s: s + "\n# BREAK\n")])
    evo.run(iterations=2)
    st = evo.state
    assert st["incumbent"] == "c001"
    assert [(h["candidate"], h["verdict"]) for h in st["history"]] == [("c001", "kept"), ("c002", "dropped")]
    h1 = st["history"][0]
    assert h1["val_child"] == pytest.approx(0.75) and h1["fixed"] == ["v1", "v2"]
    assert st["history"][1]["broke"] == ["v4"]
    # the board saw the seed and each iteration, with spend
    kinds = [e["kind"] for e in board.events]
    assert kinds == ["seed", "iteration", "iteration"]
    assert board.events[1]["accepted"] is True and board.events[1]["title"] == "Fix it" and board.events[1]["spend_usd"] > 0
    assert "Fix it" in "".join(out)


def test_seed_is_calibrated_with_two_validation_trials_and_the_margin_follows(tmp_path):
    evo, ev, *_ = make(tmp_path, [])
    evo.run(iterations=0)
    seed_spec = ev.specs[0]
    assert {"candidate": "c000", "tasks": SPLITS["validation"], "trials": 2} in seed_spec["runs"]
    assert {"candidate": "c000", "tasks": SPLITS["search"], "trials": 1} in seed_spec["runs"]
    assert evo.state["margin"] == pytest.approx(0.25)  # no flips in the fake -> one task of four


def test_the_parent_is_remeasured_in_the_same_job_as_the_child(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1)
    runs = ev.specs[1]["runs"]
    assert {"candidate": "c001", "tasks": SPLITS["search"] + SPLITS["validation"], "trials": 1} in runs
    assert {"candidate": "c000", "tasks": SPLITS["validation"], "trials": 1} in runs


def test_the_proposer_sees_search_traces_of_the_incumbent_and_the_history_but_no_validation_traces(tmp_path):
    evo, ev, pr, *_ = make(tmp_path, [("Fix it", add_fix), ("Again", lambda s: s + "\n# again\n")])
    evo.run(iterations=2)
    files = pr.seen[1]["files"]
    assert "agent.py" in files and "CONTRACT.md" in files and "PROMPT.md" in files and "traces/INDEX.md" in files
    assert any(f.startswith("traces/s1") for f in files)
    assert not any(f.startswith("traces/v") for f in files)
    assert "Fix it" in pr.seen[1]["history"] and "kept" in pr.seen[1]["history"]
    assert "# FIX" in (evo.run_dir / "candidates" / "c001" / "agent.py").read_text()


def test_an_audit_failure_or_a_crashing_candidate_costs_no_evaluation(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Use os", lambda s: "import os\n" + s), ("Crash", lambda s: s.replace("def run(task, env):", "def run(task, env):\n    raise KeyError('boom')"))])
    evo.run(iterations=2)
    assert len(ev.specs) == 1  # only the seed calibration
    verdicts = [h["verdict"] for h in evo.state["history"]]
    assert verdicts == ["rejected: audit", "rejected: crashed in preflight"]
    assert "KeyError" in evo.state["history"][1]["detail"]


def test_an_unchanged_agent_is_skipped(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Nothing", None)])
    evo.run(iterations=1)
    assert evo.state["history"][0]["verdict"] == "skipped: agent.py unchanged" and len(ev.specs) == 1


def test_spend_is_shown_every_iteration_and_kept_in_state(tmp_path):
    evo, ev, pr, board, out = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1)
    n_trials = 2 + 4 * 2 + (2 + 4) + 4 + (4 + 4)  # seed calibration, child search+val and parent val, re-measure of both
    assert evo.state["spend_usd"] == pytest.approx(n_trials * 0.003 + 3 * 0.0004)
    assert sum("spent so far" in o for o in out) >= 2


def test_heldout_check_compares_seed_and_best(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1, heldout_trials=2)
    spec = ev.specs[-1]
    assert {"candidate": "c000", "tasks": SPLITS["heldout"], "trials": 2} in spec["runs"]
    assert {"candidate": "c001", "tasks": SPLITS["heldout"], "trials": 2} in spec["runs"]
    ho = evo.state["heldout"]
    assert {k: ho[k] for k in ("seed", "best", "best_candidate", "n_tasks", "trials")} == {"seed": pytest.approx(0.5), "best": pytest.approx(1.0),
                                                                                         "best_candidate": "c001", "n_tasks": 2, "trials": 2}


def test_an_evaluation_job_that_fails_skips_the_iteration_instead_of_ending_the_run(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix), ("Fix again", lambda s: s + "\n# FIX\n# again\n")])
    real = evo.evaluate
    calls = []

    def flaky(spec, on_record):
        calls.append(1)
        if len(calls) == 2:
            raise launch.EvalJobError("Job timeout")
        return real(spec, on_record)

    evo.evaluate = flaky
    evo.run(iterations=2)
    assert [h["verdict"] for h in evo.state["history"]] == ["skipped: evaluation failed", "kept"]


def test_a_crashed_run_resumes_from_its_folder_without_repeating_paid_work(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1)
    ev2, pr2 = FakeEval(), FakeProposer([("Again", lambda s: s + "\n# again\n")])
    evo2 = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=ev2, propose=pr2, write=lambda s: None, budget_s=150, resume=True)
    evo2.run(iterations=2)
    assert len(ev2.specs) == 1  # no second seed calibration, one new iteration
    assert [h["candidate"] for h in evo2.state["history"]] == ["c001", "c002"]
    assert evo2.state["incumbent"] in ("c001", "c002") and evo2.state["spend_usd"] > evo.state["spend_usd"]


def test_validation_trials_per_task_are_configurable(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)], val_trials=2)
    evo.run(iterations=1)
    runs = ev.specs[1]["runs"]
    assert {"candidate": "c001", "tasks": SPLITS["search"], "trials": 1} in runs
    assert {"candidate": "c001", "tasks": SPLITS["validation"], "trials": 2} in runs
    assert {"candidate": "c000", "tasks": SPLITS["validation"], "trials": 2} in runs


def test_everything_lands_in_the_run_folder(tmp_path):
    evo, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1)
    d = evo.run_dir
    assert (d / "config.json").exists() and (d / "state.json").exists() and (d / "transcript.jsonl").exists()
    assert (d / "candidates" / "c001" / "diff.patch").read_text().count("+# FIX") == 1
    recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines()]
    assert {r["eval"] for r in recs} == {"e000", "e001", "e001r"} and all("split" in r for r in recs)


# --- Phase 2: keep rule with z, a re-measure before a patch counts, cost and time per iteration -------------------

class CountingEval(FakeEval):
    """Like FakeEval, but the outcome may depend on how many Jobs ran before (a lucky first measurement)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __call__(self, spec, on_record):
        n = len(self.specs)
        global outcome
        saved = outcome
        outcome = lambda source, task, trial: self.fn(source, task, trial, n)  # noqa: E731
        try:
            return super().__call__(spec, on_record)
        finally:
            outcome = saved


def lucky_once(source, task, trial, n_job):
    """'# LUCKY' passes v1 in the first Job that measures it (Job 1) and is no better than the seed afterwards."""
    if "# LUCKY" in source and n_job == 1 and task == "v1":
        return "pass"
    return saved_outcome(source, task, trial)


saved_outcome = outcome


def test_a_kept_patch_is_remeasured_once_and_dropped_when_the_win_does_not_hold(tmp_path):
    ev = CountingEval(lucky_once)
    evo, _, pr, board, out = make(tmp_path, [("Lucky", lambda s: s + "\n# LUCKY\n")])
    evo.evaluate = ev
    evo.run(iterations=1)
    h = evo.state["history"][0]
    assert h["verdict"] == "dropped: did not hold on re-measure"
    assert evo.state["incumbent"] == "c000"
    confirm = ev.specs[-1]["runs"]
    assert {"candidate": "c001", "tasks": SPLITS["validation"], "trials": 1} in confirm
    assert {"candidate": "c000", "tasks": SPLITS["validation"], "trials": 1} in confirm
    assert h["val_child_first"] == pytest.approx(0.5) and h["val_child"] == pytest.approx(0.375)  # pooled over both Jobs


def test_a_patch_that_holds_on_the_remeasure_is_kept(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1)
    assert len(ev.specs) == 3  # seed, iteration, re-measure
    assert evo.state["history"][0]["verdict"] == "kept" and evo.state["incumbent"] == "c001"


def test_the_remeasure_can_be_switched_off(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)], confirm=False)
    evo.run(iterations=1)
    assert len(ev.specs) == 2 and evo.state["history"][0]["verdict"] == "kept"


def flaky_seed(source, task, trial, n_job):
    """The seed passes v3 on its first trial only: one flip in four tasks."""
    if task == "v3" and "# FIX" not in source:
        return "pass" if trial == 0 else "fail"
    return saved_outcome(source, task, trial)


def test_the_keep_margin_uses_z(tmp_path):
    import math

    evo, *_ = make(tmp_path, [], z=2.0)
    evo.evaluate = CountingEval(flaky_seed)
    evo.run(iterations=0)
    assert evo.state["flip_rate"] == pytest.approx(0.25)
    assert evo.state["margin"] == pytest.approx(2.0 * math.sqrt(0.25 / 4))
    assert json.loads((evo.run_dir / "config.json").read_text())["strategy"]["z"] == 2.0


def test_each_iteration_records_its_time_and_cost(tmp_path):
    evo, *_ = make(tmp_path, [("Fix it", add_fix), ("Nothing", None)])
    evo.run(iterations=2)
    h1, h2 = evo.state["history"]
    assert h1["iter_usd"] > 0 and h1["iter_s"] >= 0
    assert h2["iter_usd"] == 0


def test_heldout_reports_a_paired_difference_with_a_confidence_interval(tmp_path):
    evo, ev, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1, heldout_trials=2)
    ho = evo.state["heldout"]
    assert ho["diff"] == pytest.approx(0.5) and ho["ci_low"] <= ho["diff"] <= ho["ci_high"]
    assert ho["per_task"] == {"h1": [0, 2, 2, 2], "h2": [2, 2, 2, 2]}  # [seed passes, seed trials, best passes, best trials]


def test_the_margin_accounts_for_validation_trials(tmp_path):
    import math

    evo, *_ = make(tmp_path, [], z=2.0, val_trials=2)
    evo.evaluate = CountingEval(flaky_seed)
    evo.run(iterations=0)
    assert evo.state["margin"] == pytest.approx(2.0 * math.sqrt(0.25 / (4 * 2)))


def test_a_resumed_run_keeps_the_budget_it_started_with(tmp_path):
    evo, *_ = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1)
    evo2 = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=FakeEval(), propose=FakeProposer([]), write=lambda s: None, budget_s=999, resume=True)
    assert evo2.budget_s == 150


def test_the_heldout_summary_shows_the_paired_interval(tmp_path):
    evo, ev, pr, board, out = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1, heldout_trials=2)
    text = "".join(out)
    assert "95% CI" in text and "paired" in text


def test_a_patch_dropped_on_remeasure_shows_its_numbers_on_the_card(tmp_path):
    out = []
    evo, *_ = make(tmp_path, [("Lucky", lambda s: s + "\n# LUCKY\n")])
    evo._write = out.append
    evo.evaluate = CountingEval(lucky_once)
    evo.run(iterations=1)
    card = "".join(out)
    import re

    assert "DROPPED (did not hold on re-measure)" in card and re.search(r"first measurement +25% -> +50%", card)
    assert re.search(r"validation +25% -> +38%", card)
