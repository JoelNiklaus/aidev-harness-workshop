"""Command line: the thin layer over the modules (each command's work is tested at its module)."""

import json

import pytest

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


def test_run_takes_a_seed_file_a_splits_file_and_a_strategy_folder(tmp_path, monkeypatch):
    from aidev_evolve import strategy

    seed = tmp_path / "seed.py"
    seed.write_text("def run(task, env):\n    return None\n")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"search": ["a"], "validation": ["b", "c"], "heldout": ["d"]}))
    d = strategy.init("baseline", tmp_path / "mine")
    (d / "strategy.toml").write_text((d / "strategy.toml").read_text().replace("z = 1.5", "z = 1.0").replace("confirm = true", "confirm = false"))
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    args = cli.build_parser().parse_args(["run", "--seed", str(seed), "--splits", str(splits), "--strategy", str(d), "--budget-s", "180"])
    evo = cli.make_evolver(args, tmp_path / "run")
    assert evo.source("c000") == seed.read_text()
    assert evo.splits["validation"] == ["b", "c"] and evo.z == 1.0 and evo.confirm is False and evo.budget_s == 180
    assert evo.strategy.modified and evo.strategy.preset == "baseline"


def test_run_defaults_baseline_strategy_fixed_heldout_and_the_session_time_box(tmp_path, monkeypatch):
    """One source of truth: the splits file carries the budget, the strategy the keep rule; the held-out trials and the
    session length are fixed for everyone and `run` has no flag for them."""
    from aidev_evolve import loop, strategy

    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    args = cli.build_parser().parse_args(["run"])
    evo = cli.make_evolver(args, tmp_path / "run")
    assert evo.budget_s == loop.default_splits()["budget_s"]
    assert evo.strategy.name == "baseline" and (evo.val_trials, evo.z, evo.confirm) == (2, 1.5, True)
    assert evo.session_s == strategy.SESSION_MINUTES * 60 and evo.parallel_proposals is True
    for flag in ("--z", "--val-trials", "--no-confirm", "--heldout-trials", "--iterations"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run", flag, "1"])


def test_run_takes_a_preset_name(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    evo = cli.make_evolver(cli.build_parser().parse_args(["run", "--strategy", "two-candidates"]), tmp_path / "run")
    assert evo.strategy.candidates == 2


def test_a_bad_strategy_stops_run_before_anything_is_spent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda: (_ for _ in ()).throw(AssertionError("no HF call")))
    assert cli.main(["run", "--strategy", "nope", "--proposer", "manual"]) == 2
    assert "baseline" in capsys.readouterr().out


def test_strategies_lists_every_preset_with_its_estimate(capsys):
    from aidev_evolve import strategy

    assert cli.main(["strategies"]) == 0
    out = capsys.readouterr().out
    for name in strategy.presets():
        assert name in out
    assert "HF credits" in out and "min" in out


def test_strategy_init_copies_a_preset(tmp_path, capsys):
    assert cli.main(["strategy-init", "small-diffs", str(tmp_path / "mine")]) == 0
    assert (tmp_path / "mine" / "proposer_prompt.md").exists()
    assert "--strategy" in capsys.readouterr().out


def test_rank_reads_run_folders_and_prints_the_strategy_table(tmp_path, capsys):
    from aidev_evolve import strategy

    for i, (name, rate) in enumerate([("baseline", 1)] * 5 + [("small-diffs", 0)] * 5):
        d = tmp_path / f"r{i}"
        d.mkdir()
        s = strategy.load(name)
        (d / "config.json").write_text(json.dumps({"run_id": f"u{i}-r", "strategy": {**s.event()}}))
        (d / "state.json").write_text(json.dumps({"run_id": f"u{i}-r", "history": [{"title": "Lift the step limit", "verdict": "kept"}],
                                                  "heldout": {"per_task": {"t1": [1, 2, 2 * rate, 2], "t2": [0, 2, 2 * rate, 2]}, "best_candidate": "c001"}}))
    assert cli.main(["rank", *[str(tmp_path / f"r{i}") for i in range(10)]]) == 0
    out = capsys.readouterr().out
    assert "Winning strategy: baseline" in out and "small-diffs" in out


def test_eval_defaults_to_the_bundled_budget(tmp_path):
    from aidev_evolve import loop

    args = cli.build_parser().parse_args(["eval", "--agent", "a.py", "--tasks", "x"])
    assert args.budget_s == loop.default_splits()["budget_s"]
