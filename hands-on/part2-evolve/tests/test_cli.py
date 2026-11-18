"""Command line: the thin layer over the modules (each command's work is tested at its module)."""

import json

from aidev_evolve import cli, launch


def test_eval_command_runs_one_candidate_and_writes_records(tmp_path, monkeypatch):
    agent = tmp_path / "agent.py"
    agent.write_text("def run(task, env):\n    pass\n")
    calls = {}

    def fake_run_eval(api, token, user, spec, on_record, **kw):
        calls["spec"] = spec
        recs = [{"candidate": "cand", "task": t, "outcome": "pass", "cost_usd": 0.01, "wall_s": 60} for t in spec["runs"][0]["tasks"]]
        for r in recs:
            on_record(r)
        return launch.EvalResult(recs, "job1", None, 90.0, 0.0003)

    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    monkeypatch.setattr(launch, "run_eval", fake_run_eval)
    out = tmp_path / "recs.jsonl"
    assert cli.main(["eval", "--agent", str(agent), "--tasks", "fix-git,regex-log", "--trials", "2", "--out", str(out)]) == 0
    assert calls["spec"]["runs"] == [{"candidate": "cand", "tasks": ["fix-git", "regex-log"], "trials": 2}]
    assert len(out.read_text().splitlines()) == 2
    assert json.loads(out.read_text().splitlines()[0])["task"] == "fix-git"


def test_run_takes_a_seed_file_a_splits_file_and_the_keep_rule(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text("def run(task, env):\n    return None\n")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"search": ["a"], "validation": ["b", "c"], "heldout": ["d"]}))
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    args = cli.build_parser().parse_args(["run", "--seed", str(seed), "--splits", str(splits), "--z", "1.5", "--no-confirm", "--budget-s", "180"])
    evo = cli.make_evolver(args, tmp_path / "run")
    assert evo.source("c000") == seed.read_text()
    assert evo.splits["validation"] == ["b", "c"] and evo.z == 1.5 and evo.confirm is False and evo.budget_s == 180


def test_run_defaults_are_the_phase2_configuration(tmp_path, monkeypatch):
    """One source of truth: the bundled splits file carries the time budget its tasks were selected for; the keep
    rule defaults to 2 validation trials, z = 1.5 and a re-measure; the held-out check uses 2 trials."""
    from aidev_evolve import loop

    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    args = cli.build_parser().parse_args(["run"])
    evo = cli.make_evolver(args, tmp_path / "run")
    assert evo.budget_s == loop.default_splits()["budget_s"]
    assert (evo.val_trials, evo.z, evo.confirm, args.heldout_trials) == (2, 1.5, True, 2)


def test_eval_defaults_to_the_bundled_budget(tmp_path):
    from aidev_evolve import loop

    args = cli.build_parser().parse_args(["eval", "--agent", "a.py", "--tasks", "x"])
    assert args.budget_s == loop.default_splits()["budget_s"]
