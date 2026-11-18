"""Scores, the noise margin and the keep-or-drop rule (harness-optimization-search: paired, re-measured parent)."""

import math

import pytest

from aidev_evolve import stats


def recs(candidate, outcomes):
    """outcomes: {task: "PPF"} -> records (P pass, F fail, I infra)."""
    m = {"P": "pass", "F": "fail", "I": "infra"}
    return [{"candidate": candidate, "task": t, "outcome": m[c], "cost_usd": 0.01} for t, s in outcomes.items() for c in s]


def test_per_task_pass_rates_ignore_infra():
    pt = stats.per_task(recs("a", {"x": "PFI", "y": "II", "z": "P"}), "a")
    assert pt == {"x": [1, 0], "z": [1]}
    assert stats.score(pt) == pytest.approx(0.75)


def test_flip_rate_from_two_trials_per_task():
    pt = {"a": [1, 1], "b": [1, 0], "c": [0, 0], "d": [0, 1]}
    assert stats.flip_rate(pt) == 0.5


def test_margin_follows_the_paired_standard_error_and_is_at_least_one_task():
    assert stats.margin(0.25, 8) == pytest.approx(max(1 / 8, math.sqrt(0.25 / 8)))
    assert stats.margin(0.0, 8) == pytest.approx(1 / 8)


def test_keep_when_the_child_beats_the_pooled_parent_by_the_margin_on_shared_tasks():
    parent = recs("p", {"a": "FF", "b": "PF", "c": "PP", "d": "FF"})
    child = recs("c", {"a": "P", "b": "P", "c": "P", "d": "F", "e": "P"})  # e: parent never ran it
    d = stats.decide(child + parent, "c", "p", margin=0.25)
    assert d.n_tasks == 4
    assert d.parent_score == pytest.approx((0 + 0.5 + 1 + 0) / 4)
    assert d.child_score == pytest.approx(3 / 4)
    assert d.delta == pytest.approx(0.375) and d.keep
    assert d.fixed == ["a", "b"] and d.broke == []


def test_drop_when_the_gain_is_inside_the_noise():
    parent = recs("p", {"a": "F", "b": "P", "c": "F", "d": "P"})
    child = recs("c", {"a": "P", "b": "P", "c": "F", "d": "P"})
    d = stats.decide(child + parent, "c", "p", margin=0.3)
    assert d.delta == pytest.approx(0.25) and not d.keep


def test_a_regression_lists_the_broken_tasks():
    parent = recs("p", {"a": "P", "b": "P"})
    child = recs("c", {"a": "F", "b": "P"})
    d = stats.decide(child + parent, "c", "p", margin=0.1)
    assert d.broke == ["a"] and not d.keep


def test_no_shared_scored_task_means_drop():
    d = stats.decide(recs("c", {"a": "P"}) + recs("p", {"b": "P"}), "c", "p", margin=0.1)
    assert not d.keep and d.n_tasks == 0


def test_paired_difference_with_a_bootstrap_interval_over_tasks():
    a = recs("seed", {"x": "FF", "y": "PP", "z": "FF", "w": "PF"})
    b = recs("best", {"x": "PP", "y": "PP", "z": "PF", "w": "PF"})
    d = stats.paired_diff(a + b, "best", "seed", ["x", "y", "z", "w"])
    assert d.diff == pytest.approx((1 + 0 + 0.5 + 0) / 4)
    assert d.ci_low <= d.diff <= d.ci_high and d.ci_low >= 0 and d.ci_high <= 1
    assert d.n_tasks == 4


def test_paired_difference_of_identical_agents_has_a_zero_width_interval():
    a = recs("seed", {"x": "PP", "y": "FF"})
    b = recs("best", {"x": "PP", "y": "FF"})
    d = stats.paired_diff(a + b, "best", "seed")
    assert (d.diff, d.ci_low, d.ci_high) == (0, 0, 0)


def test_margin_shrinks_with_trials_per_task():
    # each side's per-task rate averages k trials: the paired difference's variance is about d / k per task
    assert stats.margin(0.4, 5, z=1.5, k=2) == pytest.approx(max(1 / 5, 1.5 * math.sqrt(0.4 / (5 * 2))))
    assert stats.margin(0.4, 5, z=1.5, k=2) < stats.margin(0.4, 5, z=1.5)
