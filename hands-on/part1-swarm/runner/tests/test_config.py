"""The workshop grid: the 20-task TB2.1 subset (config/tb21-subset.json), harnesses in Joel's order."""

import json
from pathlib import Path

CONFIG = Path(__file__).parents[2] / "config"


def load(name):
    return json.loads((CONFIG / name).read_text())


def test_grid_tasks_are_exactly_the_subset_tasks_in_subset_order():
    grid, subset = load("grid.json"), load("tb21-subset.json")
    assert grid["tasks"] == [t["task"] for t in subset["tasks"]]
    assert len(grid["tasks"]) == 20


def test_the_20_task_subset_keeps_the_17_dry_run_tasks_and_their_15_minute_cap():
    # the 20-Job dry run and the first reference layer covered the 17; the 3 additions keep the native 15-min timeout
    subset = load("tb21-subset.json")
    assert subset["name"] == "tb21-workshop-20"
    added = {"fix-git", "polyglot-c-py", "regex-log"}
    assert added <= {t["task"] for t in subset["tasks"]}
    assert all(t["agent_timeout_s"] <= 900 for t in subset["tasks"])


def test_grid_uses_the_subset_model_and_benchmark():
    grid, subset = load("grid.json"), load("tb21-subset.json")
    assert grid["model"] == subset["model"] and grid["benchmark"] == subset["benchmark"]


def test_reserve_tasks_stay_out_of_the_grid():
    grid, subset = load("grid.json"), load("tb21-subset.json")
    assert not {r["task"] for r in subset["reserve_tasks_one_rule_missed"]} & set(grid["tasks"])


def test_harness_order_pi_opencode_openclaw_first_then_the_other_seven():
    h = load("grid.json")["harnesses"]
    assert h[:3] == ["pi", "opencode", "openclaw"]
    assert len(h) == len(set(h)) == 10


def test_effort_low_and_three_trials_per_cell():
    grid = load("grid.json")
    assert grid["reasoning_effort"] == "low" and grid["trials_per_cell"] == 3
