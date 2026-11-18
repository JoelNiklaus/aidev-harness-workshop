"""Part 2 strategy ranking on /evolve (part2-evolve/REQUESTS.md, PROTOCOL.md "Strategy fields"): the strategy block,
siblings, held-out per-task counts and kept titles survive validation; the state ranks strategies with the vendored
ranking.py, by preset and by hash, and names the winning strategy and patch."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from swarm_coordinator import evolve, ranking
from swarm_coordinator.store import Invalid

from conftest import Clock
from test_evolve import heldout, it, seed

TASKS = [f"t{i:02d}" for i in range(23)]
KNOBS = {"traces": "failed", "max_traces": 4, "history": "titles", "diffs": False, "candidates": 1, "val_trials": 2, "z": 1.5,
         "confirm": True, "iterations": 4}


def strat(preset="baseline", name=None, hash_=None, modified=False, prompt="Read the traces, then fix the most common failure."):
    return {"name": name or preset, "preset": preset, "hash": hash_ or hashlib.sha256(preset.encode()).hexdigest()[:10], "modified": modified,
            "prompt": prompt, "knobs": dict(KNOBS)}


def per_task(rate, rng, seed_rate=0.5):
    return {t: [sum(rng.random() < seed_rate for _ in range(2)), 2, sum(rng.random() < rate for _ in range(2)), 2] for t in TASKS}


def full_run(user, preset, rate, *, rng, kept_title="Use the time budget, not 10 steps", provider="novita", **skw):
    """The events one attendee's run posts: seed, a kept iteration, a dropped one, held-out with per-task counts."""
    s, rid = strat(preset, **skw), f"{user}-{preset}"
    return [(user, {**seed(rid), "strategy": s, "provider": provider}),
            (user, {**it(1, run_id=rid, accepted=True, best=0.8, title=kept_title), "strategy": s, "provider": provider,
                    "siblings": [{"candidate": "c001a", "title": "Show stderr", "val": 0.5, "verdict": "dropped: lost the screen"}]}),
            (user, {**it(2, run_id=rid, best=0.8, title="Verify before finishing"), "strategy": s, "provider": provider}),
            (user, {**heldout(rid), "strategy": s, "provider": provider, "per_task": per_task(rate, rng), "kept": [kept_title]})]


def store_with(posts):
    st = evolve.EvolveStore(clock=Clock())
    for user, e in posts:
        st.add(user, evolve.validate(e))
    return st


# ---- validation -------------------------------------------------------------------------------------


def test_strategy_block_is_kept_on_every_kind():
    s = strat("small-diffs", modified=True)
    for e in (seed(), it(1, accepted=True, best=0.8), heldout()):
        out = evolve.validate({**e, "strategy": s})
        assert out["strategy"] == s


def test_strategy_text_is_cut_and_unknown_knobs_dropped():
    s = {**strat(), "name": "n" * 100, "prompt": "p" * 5000, "knobs": {**KNOBS, "secret": "x"}}
    out = evolve.validate({**seed(), "strategy": s})["strategy"]
    assert len(out["name"]) == 60 and len(out["prompt"]) == 4000 and "secret" not in out["knobs"]


@pytest.mark.parametrize("bad", [{"hash": 12}, {"hash": "not hex!"}, {"modified": "yes"}, {"knobs": {"candidates": 7}},
                                 {"knobs": {"traces": "some"}}, {"name": None}, "baseline"])
def test_bad_strategy_blocks_are_refused(bad):
    s = bad if isinstance(bad, str) else {**strat(), **bad}
    with pytest.raises(Invalid):
        evolve.validate({**seed(), "strategy": s})


def test_siblings_per_task_kept_and_provider_survive():
    rng = random.Random(1)
    sib = [{"candidate": "c001a", "title": "Show stderr", "val": 0.5, "verdict": "dropped: lost the screen"}]
    out = evolve.validate({**it(1, accepted=True, best=0.8), "siblings": sib, "provider": "novita"})
    assert out["siblings"] == sib and out["provider"] == "novita"
    pt = per_task(0.7, rng)
    out = evolve.validate({**heldout(), "per_task": pt, "kept": ["Use the time budget"]})
    assert out["per_task"] == pt and out["kept"] == ["Use the time budget"]


def test_per_task_is_capped_at_40_tasks_and_must_be_counts():
    big = {f"task-{i}": [1, 2, 2, 2] for i in range(60)}
    assert len(evolve.validate({**heldout(), "per_task": big})["per_task"]) == 40
    for bad in ({"t": [3, 2, 0, 2]}, {"t": [1, 2]}, {"t": [1, 2, -1, 2]}, {"t": "x"}, ["t"]):
        with pytest.raises(Invalid):
            evolve.validate({**heldout(), "per_task": bad})


def test_events_without_strategy_fields_still_validate():
    # 0.3.0 clients send none of them
    assert "strategy" not in evolve.validate(seed()) and "per_task" not in evolve.validate(heldout())


def test_iteration_events_are_stored_without_the_prompt_text():
    # the prompt (up to 4,000 chars) rides on every event; the board needs it once per run
    st = store_with(full_run("alice", "baseline", 0.8, rng=random.Random(2), prompt="P" * 3000))
    by_kind = {e["kind"]: e for e in st.events()}
    assert "prompt" not in by_kind["iteration"]["strategy"] and by_kind["seed"]["strategy"]["prompt"] == "P" * 3000


# ---- ranking in the state ----------------------------------------------------------------------------


def room(n_per_preset=6, rates=None):
    rng = random.Random(7)
    rates = rates or {"baseline": 0.8, "failures-first": 0.8, "small-diffs": 0.5, "two-candidates": 0.8}
    posts = []
    for preset, rate in rates.items():
        for i in range(n_per_preset):
            posts += full_run(f"{preset[:4]}{i}", preset, rate, rng=rng)
    return posts


def test_vendored_ranking_is_the_part2_file_unchanged():
    part2 = Path(__file__).parents[3] / "part2-evolve" / "src" / "aidev_evolve" / "ranking.py"
    if not part2.exists():
        pytest.skip("part2-evolve not in this checkout")
    assert Path(ranking.__file__).read_text() == part2.read_text()


def test_state_ranks_presets_and_hashes_with_the_vendored_ranking():
    st = store_with(room())
    s = evolve.build_state(st.events(), [])
    runs = evolve.ranking_runs(st.events())
    assert len(runs) == 24 and {"run_id", "user", "strategy", "heldout", "kept"} <= set(runs[0])
    want = ranking.rank(runs, by="preset")
    assert [e["key"] for e in s["presets"]["strategies"]] == [e["key"] for e in want["strategies"]]
    assert s["presets"]["groups"] == want["groups"]
    assert s["strategies"]["by"] == "hash" and s["presets"]["by"] == "preset" and s["min_runs"] == ranking.MIN_RUNS
    small = next(e for e in s["presets"]["strategies"] if e["key"] == "small-diffs")
    assert small["rank"] > 1  # 30 points behind: not tied with the leaders


def test_winner_names_its_strategy_knobs_prompt_and_the_patch_iterations_with_diffs():
    st = store_with(room())
    w = evolve.build_state(st.events(), [])["presets"]["winner"]
    assert w["keys"] and w["strategies"][0]["knobs"] == KNOBS and w["strategies"][0]["prompt"].startswith("Read the traces")
    p = w["patch"]
    assert p["kept"] == ["Use the time budget, not 10 steps"]
    assert p["iterations"] == [{"iteration": 1, "title": "Use the time budget, not 10 steps",
                                "diff_url": f"api/evolve/diff/{p['user']}/{p['run_id']}/1"}]


def test_too_few_runs_are_listed_not_ranked_and_there_is_no_winner():
    st = store_with(room(n_per_preset=2))
    pr = evolve.build_state(st.events(), [])["presets"]
    assert pr["winner"] is None and all(not e["ranked"] and "needs 5 runs" in e["note"] for e in pr["strategies"])


def test_no_heldout_yet_gives_empty_rankings():
    st = store_with(room()[:2])
    s = evolve.build_state(st.events(), [])
    assert s["presets"]["strategies"] == [] and s["strategies"]["winner"] is None


def test_runs_carry_strategy_name_preset_and_provider():
    st = store_with(full_run("alice", "small-diffs", 0.7, rng=random.Random(3)))
    [r] = evolve.build_state(st.events(), [])["runs"]
    assert (r["strategy"], r["preset"], r["provider"]) == ("small-diffs", "small-diffs", "novita")


def test_rank_cache_recomputes_only_when_heldouts_change_and_waits_ten_times_a_slow_ranking():
    calls = []
    clock = Clock()

    def slow_rank(runs, by):  # 1.5 s per call, as with ~400 runs: one rebuild ranks twice (hash, preset) = 3 s
        calls.append(by)
        clock.advance(1.5)
        return ranking.rank(runs, by=by)

    cache = evolve.RankCache(clock=clock, rank=slow_rank)
    events = store_with(room(n_per_preset=5)).events()
    first = evolve.build_state(events, [], rank_cache=cache)
    evolve.build_state(events, [], rank_cache=cache)
    assert calls == ["hash", "preset"]  # unchanged held-outs: cached
    more = store_with(room(n_per_preset=6)).events()
    clock.advance(5)
    assert evolve.build_state(more, [], rank_cache=cache)["presets"] == first["presets"]  # changed, but 3 s of ranking buys 30 s
    clock.advance(30)
    assert evolve.build_state(more, [], rank_cache=cache)["presets"]["n_runs"] == 24 and len(calls) == 4


def test_rank_cache_reranks_at_once_while_ranking_is_cheap():
    clock = Clock()
    cache = evolve.RankCache(clock=clock)
    evolve.build_state(store_with(room(n_per_preset=2)).events(), [], rank_cache=cache)
    assert evolve.build_state(store_with(room(n_per_preset=3)).events(), [], rank_cache=cache)["presets"]["n_runs"] == 12
