"""Room conveniences (0.5.3): strategies by seating block (`run --block`), the final agent.py easy to open
(`final_agent.py` in the run folder, its path printed, `aidev-evolve show`), and the pin shared with Part 1."""

import json
import re
from pathlib import Path

import pytest

from aidev_evolve import cli, loop, show, strategy

from test_loop import add_fix, make

ROOT = Path(__file__).parents[1]


# -- the pin -----------------------------------------------------------------------------------------

def _hlb_pin(pyproject: Path) -> str:
    return re.search(r"harness-leaderboard\[worker\] @ git\+[^@]+@([0-9a-f]{40})", pyproject.read_text()).group(1)


def test_the_evaluation_pins_the_same_leaderboard_revision_as_the_part1_runner():
    """One pin for both hands-ons: the revision that stops an agent at its timeout (test_agent_stop.py)."""
    assert _hlb_pin(ROOT / "pyproject.toml") == _hlb_pin(ROOT.parent / "part1-swarm" / "runner" / "pyproject.toml")
    assert _hlb_pin(ROOT / "pyproject.toml") in (ROOT / "uv.lock").read_text()


# -- seating blocks ------------------------------------------------------------------------------------

def test_each_seating_block_has_one_preset_and_the_three_differ():
    assert strategy.BLOCKS == {"left": "baseline", "middle": "every-weakness", "right": "code-audit"}
    assert set(strategy.BLOCKS.values()) <= set(strategy.presets())


@pytest.mark.parametrize(("flag", "preset"), [("left", "baseline"), ("middle", "every-weakness"), ("right", "code-audit"),
                                              ("A", "baseline"), ("b", "every-weakness"), ("C", "code-audit"), ("LEFT", "baseline")])
def test_run_block_picks_the_blocks_strategy(flag, preset):
    args = cli.build_parser().parse_args(["run", "--block", flag])
    assert args.strategy == preset


def test_run_without_block_keeps_the_default_strategy_and_an_explicit_one_wins():
    p = cli.build_parser()
    assert p.parse_args(["run"]).strategy == strategy.DEFAULT
    assert p.parse_args(["run", "--strategy", "small-diffs"]).strategy == "small-diffs"


def test_block_and_strategy_together_are_refused(capsys):
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(["run", "--block", "left", "--strategy", "code-audit"])
    assert e.value.code == 2 and "not allowed with" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--block", "balcony"])
    assert "left, middle, right" in capsys.readouterr().err  # the error names the blocks


def test_strategies_lists_the_block_of_each_assigned_preset(capsys):
    assert cli.main(["strategies"]) == 0
    out = capsys.readouterr().out
    for block, preset in strategy.BLOCKS.items():
        assert f"--block {block}" in out and re.search(rf"--block {block}\b.*\b{preset}\b", out)


# -- the final agent ------------------------------------------------------------------------------------

def test_a_finished_run_leaves_final_agent_py_and_says_where(tmp_path):
    evo, *_, out = make(tmp_path, [("Fix it", add_fix)])
    evo.run(iterations=1, heldout_trials=1)
    final = tmp_path / "run" / "final_agent.py"
    assert final.read_text() == evo.source("c001") and "# FIX" in final.read_text()
    text = "".join(out)
    assert str(final) in text and text.index(str(final)) > text.index("Held-out check")


def test_a_run_that_kept_nothing_leaves_the_seed_as_final_agent(tmp_path):
    evo, *_ = make(tmp_path, [("No-op", None)])
    evo.run(iterations=1)
    assert (tmp_path / "run" / "final_agent.py").read_text() == evo.source("c000")


def _finished_run(tmp_path, name, edits):
    evo, *_ = make(tmp_path / name, edits)
    evo.run(iterations=len(edits), heldout_trials=1)
    return tmp_path / name / "run"


def test_show_prints_the_final_agent_its_path_and_the_kept_patches(tmp_path, capsys):
    run_dir = _finished_run(tmp_path, "a", [("Fix it", add_fix)])
    assert cli.main(["show", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert str(run_dir / "final_agent.py") in out and "Fix it" in out
    assert "def run(task, env):" in out and "# FIX" in out


def test_show_diff_prints_the_changes_against_the_seed(tmp_path, capsys):
    run_dir = _finished_run(tmp_path, "a", [("Fix it", add_fix)])
    assert cli.main(["show", str(run_dir), "--diff"]) == 0
    out = capsys.readouterr().out
    assert "+# FIX" in out and "def run(task, env):" not in out.replace(" def run(task, env):", "")


def test_show_without_a_folder_takes_the_newest_run_under_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name, title in (("20261201-100000", "Old patch"), ("20261201-103000", "New patch")):
        evo, *_ = make(tmp_path / "x" / name, [(title, add_fix)])
        evo.run(iterations=1)
        (tmp_path / "runs").mkdir(exist_ok=True)
        (tmp_path / "x" / name / "run").rename(tmp_path / "runs" / name)
    assert cli.main(["show"]) == 0
    out = capsys.readouterr().out
    assert "New patch" in out and "Old patch" not in out


def test_show_says_what_to_do_when_there_is_no_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["show"]) == 1
    assert "aidev-evolve run" in capsys.readouterr().out


def test_show_reports_which_seed_weaknesses_are_still_in_the_final_agent():
    """The debrief question "is the cleanup line still in your agent?" answered from the file itself."""
    seed = loop.SEED_SOURCE  # a seed edit that renames one of the three must fail here, not report "removed" in the room
    assert show.seed_lines_left(seed) == {"step limit": True, "task text cut": True, "stray-process cleanup": True}
    fixed = seed.replace("MAX_STEPS = 10", "MAX_STEPS = 200").replace(" + CLEANUP", "")  # the constant stays, unused
    assert show.seed_lines_left(fixed) == {"step limit": False, "task text cut": True, "stray-process cleanup": False}
    assert show.seed_lines_left("def run(task, env):\n    return None\n") == {"step limit": False, "task text cut": False, "stray-process cleanup": False}


def test_show_works_on_the_bundled_recorded_run(capsys):
    """Runs made before 0.5.3 have no final_agent.py: `show` reads the incumbent's candidate folder."""
    from aidev_evolve import replay

    assert cli.main(["show", str(replay.bundled())]) == 0
    out = capsys.readouterr().out
    state = json.loads((replay.bundled() / "state.json").read_text())
    assert f"candidates/{state['incumbent']}/agent.py" in out
    assert "stray-process cleanup: removed" in out  # the recorded code-audit run removed the hidden weakness


def test_show_diff_is_never_cut_however_long_the_change_is(tmp_path, capsys):
    """The run's own output cuts a long diff at 60 lines and points to the run folder; `show --diff` is that folder's reader."""
    many = lambda src: src + "".join(f"# FIX line {i}\n" for i in range(90))  # noqa: E731
    evo, *_ = make(tmp_path, [("Long patch", many)])
    evo.run(iterations=1)
    assert cli.main(["show", str(tmp_path / "run"), "--diff"]) == 0
    out = capsys.readouterr().out
    assert "+# FIX line 89" in out and "more diff lines" not in out
