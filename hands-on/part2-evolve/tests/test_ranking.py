"""Ranking strategies by held-out gain: pooled per strategy, paired bootstrap over runs, leaderboard tie groups."""

import random

import pytest

from aidev_evolve import ranking

TASKS = [f"t{i}" for i in range(12)]


def run(run_id, strat, final_rates, seed_rate=0.5, preset=None, trials=2, rng=None, kept=("Lift the step limit",)):
    """A run's held-out result as the board stores it: per task [seed passes, seed n, final passes, final n]."""
    rng = rng or random.Random(run_id)
    per_task = {}
    for t, p in zip(TASKS, final_rates):
        per_task[t] = [sum(rng.random() < seed_rate for _ in range(trials)), trials, sum(rng.random() < p for _ in range(trials)), trials]
    return {"run_id": run_id, "user": run_id.split("-")[0], "strategy": {"name": strat, "preset": preset or strat, "hash": f"h-{strat}"},
            "heldout": {"per_task": per_task, "best_candidate": "c001"}, "kept": list(kept)}


def many(strat, rate, n, **kw):
    return [run(f"{strat}{i}-r", strat, [rate] * len(TASKS), **kw) for i in range(n)]


def test_a_clearly_better_strategy_wins_and_equal_ones_tie():
    runs = many("good", 0.9, 8) + many("same-a", 0.5, 8) + many("same-b", 0.5, 8)
    out = ranking.rank(runs)
    order = [s["key"] for s in out["strategies"] if s["ranked"]]
    assert order[0] == "h-good"
    assert out["groups"][0] == ["h-good"]
    assert set(out["groups"][1]) == {"h-same-a", "h-same-b"}
    assert out["winner"]["keys"] == ["h-good"] and out["winner"]["tie"] is False
    good = out["strategies"][0]
    assert good["rank"] == 1 and good["n_runs"] == 8
    assert good["gain"] == pytest.approx(good["score"] - out["seed"])
    lo, hi = good["ci"]
    assert lo <= good["gain"] <= hi and lo > 0


def test_strategies_with_too_few_runs_are_listed_but_not_ranked():
    runs = many("good", 0.9, 6) + many("lonely", 1.0, 2)
    out = ranking.rank(runs, min_runs=5)
    lonely = next(s for s in out["strategies"] if s["key"] == "h-lonely")
    assert lonely["ranked"] is False and lonely["rank"] is None and "needs 5 runs" in lonely["note"]
    assert out["groups"] == [["h-good"]]


def test_a_tie_at_the_top_is_called_a_tie_not_a_win():
    runs = many("a", 0.7, 6) + many("b", 0.7, 6)
    out = ranking.rank(runs)
    assert out["winner"]["tie"] is True and set(out["winner"]["keys"]) == {"h-a", "h-b"}
    assert all(s["rank"] == 1 for s in out["strategies"])


def test_tie_groups_follow_the_leaderboard_rule_head_first():
    # tied(x, head): a ties b, b ties c, a does not tie c -> [[a, b], [c]]
    tied = {("b", "a"), ("c", "b")}
    assert ranking.tie_groups(["a", "b", "c"], lambda x, head: (x, head) in tied) == [["a", "b"], ["c"]]
    assert ranking.tie_ranks([["a", "b"], ["c"]]) == [1, 3]


def test_grouping_by_preset_pools_tweaked_copies_with_their_preset():
    runs = many("small-diffs", 0.8, 3) + [run(f"t{i}-r", "my-tweak", [0.8] * 12, preset="small-diffs") for i in range(3)] + many("baseline", 0.5, 6)
    by_hash = ranking.rank(runs, min_runs=5)
    assert not any(s["ranked"] for s in by_hash["strategies"] if s["key"] != "h-baseline")
    by_preset = ranking.rank(runs, by="preset", min_runs=5)
    sd = next(s for s in by_preset["strategies"] if s["key"] == "small-diffs")
    assert sd["ranked"] and sd["n_runs"] == 6


def test_the_winning_patch_is_the_best_run_of_the_winning_group_with_its_own_interval():
    runs = many("good", 0.85, 6) + many("bad", 0.4, 6)
    runs[2]["kept"] = ["Kill hung commands after 60 s", "Lift the step limit"]
    runs[2]["heldout"]["per_task"] = {t: [1, 2, 2, 2] for t in TASKS}
    out = ranking.rank(runs)
    wp = out["winner"]["patch"]
    assert wp["run_id"] == runs[2]["run_id"] and wp["kept"] == ["Kill hung commands after 60 s", "Lift the step limit"]
    assert wp["gain"] == pytest.approx(0.5) and wp["ci"][0] <= wp["gain"] <= wp["ci"][1]


def test_ranking_is_deterministic_and_has_a_task_resampled_interval_too():
    runs = many("good", 0.9, 6) + many("bad", 0.5, 6)
    a, b = ranking.rank(runs), ranking.rank(runs)
    assert a == b
    g = a["strategies"][0]
    assert g["ci_tasks"][0] <= g["ci"][0] and g["ci_tasks"][1] >= g["ci"][1]  # resampling tasks as well only widens it


def test_runs_without_a_heldout_result_are_ignored():
    runs = many("good", 0.9, 5) + [{"run_id": "x-r", "user": "x", "strategy": {"name": "good", "preset": "good", "hash": "h-good"}, "heldout": None, "kept": []}]
    out = ranking.rank(runs)
    assert out["strategies"][0]["n_runs"] == 5


def test_empty_room_has_no_winner():
    out = ranking.rank([])
    assert out["strategies"] == [] and out["winner"] is None


def test_the_run_interval_is_a_t_interval_with_small_sample_widths():
    assert [round(ranking.t975(df), 2) for df in (4, 8, 20)] == [2.78, 2.31, 2.09]  # exact: 2.776, 2.306, 2.086
    lo, hi = ranking.welch([0.5, 0.6, 0.7, 0.8, 0.9])
    assert (lo + hi) / 2 == pytest.approx(0.7) and (hi - lo) / 2 == pytest.approx(2.776 * (0.025 ** 0.5) / (5 ** 0.5), rel=1e-2)
    assert ranking.welch([0.5, 0.5], [0.5, 0.5]) == [0.0, 0.0]


def test_a_single_run_has_no_run_interval():
    out = ranking.rank(many("solo", 0.8, 1) + many("good", 0.9, 5))
    solo = next(e for e in out["strategies"] if e["key"] == "h-solo")
    assert solo["ci"] is None and solo["ci_tasks"] is not None
    assert "[n/a]" in ranking.format_table(out)
