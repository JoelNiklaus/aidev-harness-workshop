"""Part 2 evolve board: event validation, the event store, the board payload and the fix tally."""

from __future__ import annotations

import pytest

from swarm_coordinator import evolve
from swarm_coordinator.store import Invalid

from conftest import Clock


def seed(run_id="alice-r1", val=0.4, spend=0.11, **kw):
    return {"schema": evolve.SCHEMA, "run_id": run_id, "kind": "seed", "val_score": val, "search_score": 0.0, "spend_usd": spend, **kw}


DIFF = "--- agent.py (c000)\n+++ agent.py (c001)\n@@ -1,3 +1,4 @@\n-MAX_STEPS = 10\n+MAX_STEPS = 200\n+STOP = 5\n x = 1\n"


def it(i, run_id="alice-r1", accepted=False, best=0.4, title="Show stderr to the model", spend=0.2, **kw):
    e = {"schema": evolve.SCHEMA, "run_id": run_id, "kind": "iteration", "iteration": i, "candidate": f"c{i:03d}", "parent": "c000",
         "title": title, "accepted": accepted, "verdict": "kept" if accepted else "dropped", "val_child": 0.8 if accepted else 0.4,
         "val_parent": 0.4, "best_val": best, "fixed": ["build-pmars"] if accepted else [], "broke": [], "diff": DIFF, "spend_usd": spend}
    e.update(kw)
    return e


def heldout(run_id="alice-r1", spend=1.2):
    return {"schema": evolve.SCHEMA, "run_id": run_id, "kind": "heldout", "seed": 0.56, "best": 0.75, "best_candidate": "c001",
            "n_tasks": 8, "trials": 2, "diff": 0.19, "ci_low": 0.02, "ci_high": 0.31, "spend_usd": spend}


# ---- validation -----------------------------------------------------------------------------------

def test_validate_accepts_the_three_kinds_the_client_sends():
    for e in (seed(), it(1, accepted=True, best=0.8), heldout()):
        assert evolve.validate(e)["kind"] == e["kind"]


def test_validate_drops_fields_outside_the_protocol():
    out = evolve.validate({**seed(), "secret": "x" * 50, "per_task": {"a": [1, 0]}})
    assert "secret" not in out and "per_task" not in out


@pytest.mark.parametrize("bad", [
    {"schema": "other/1"},
    {"kind": "nonsense"},
    {"run_id": ""},
    {"run_id": "a/b"},
    {"run_id": "x" * 200},
    {"spend_usd": -1},
    {"spend_usd": "free"},
    {"val_score": 1.5},
    {"val_score": True},
])
def test_validate_rejects_bad_seed_events(bad):
    with pytest.raises(Invalid):
        evolve.validate({**seed(), **bad})


@pytest.mark.parametrize("bad", [
    {"iteration": 0},
    {"iteration": 10_000},
    {"iteration": "3"},
    {"accepted": "yes"},
    {"best_val": None},
    {"title": 42},
    {"fixed": "build-pmars"},
    {"fixed": [3]},
])
def test_validate_rejects_bad_iteration_events(bad):
    with pytest.raises(Invalid):
        evolve.validate({**it(1), **bad})


def test_validate_shortens_overlong_text_instead_of_rejecting_it():
    # the client drops an event the board refuses with a 4xx, and its point would vanish from the race:
    # a long title or diff is the attendee's coding agent being verbose, not a bad request
    out = evolve.validate({**it(1), "title": "t" * 1000, "diff": "d" * 9000, "fixed": ["x" * 300] * 500})
    assert len(out["title"]) == 200 and out["title"].endswith("…")
    assert len(out["diff"]) == 8000
    assert len(out["fixed"]) == 100 and len(out["fixed"][0]) == 100


def test_validate_rejects_non_objects():
    with pytest.raises(Invalid):
        evolve.validate(["not", "an", "event"])


# ---- store ----------------------------------------------------------------------------------------

class SinkSpy:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def put_path(self, path, data):
        self.files[path] = data

    def load_prefix(self, prefix):
        return {p: d for p, d in self.files.items() if p.startswith(prefix)}


def test_store_is_idempotent_per_user_run_kind_iteration():
    s = evolve.EvolveStore(clock=Clock())
    s.add("alice", evolve.validate(it(1, title="first")))
    s.add("alice", evolve.validate(it(1, title="retried")))
    s.add("bob", evolve.validate(it(1, title="bob's")))  # same run_id, other user: a different run
    titles = sorted(e["title"] for e in s.events())
    assert titles == ["bob's", "retried"]


def test_store_caps_runs_per_user():
    s = evolve.EvolveStore(clock=Clock(), max_runs_per_user=2)
    s.add("alice", evolve.validate(seed("alice-a")))
    s.add("alice", evolve.validate(seed("alice-b")))
    s.add("alice", evolve.validate(it(1, run_id="alice-b")))  # an existing run keeps working
    with pytest.raises(Invalid):
        s.add("alice", evolve.validate(seed("alice-c")))


def test_store_writes_each_event_to_the_sink_and_restores_from_it():
    sink = SinkSpy()
    s = evolve.EvolveStore(clock=Clock(), sink=sink)
    s.add("alice", evolve.validate(seed()))
    s.add("alice", evolve.validate(it(1, accepted=True, best=0.8)))
    assert sorted(sink.files) == ["evolve/alice/alice-r1/iteration-001.json", "evolve/alice/alice-r1/seed-000.json"]
    again = evolve.EvolveStore(clock=Clock(), sink=sink)
    assert again.restore() == 2
    assert [e["kind"] for e in again.events()] == ["seed", "iteration"]
    assert again.diff("alice", "alice-r1", 1) == DIFF


def test_diff_lookup_misses_cleanly():
    s = evolve.EvolveStore(clock=Clock())
    assert s.diff("alice", "nope", 1) is None


# ---- rate limit -----------------------------------------------------------------------------------

def test_rate_limiter_allows_a_burst_then_refills():
    clock = Clock()
    rl = evolve.RateLimiter(per_minute=6, burst=3, clock=clock)
    assert [rl.allow("alice") for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = rl.allow("alice")
    assert wait == pytest.approx(10.0)  # one token every 10 s
    assert rl.allow("bob") == 0.0  # per user
    clock.advance(10.0)
    assert rl.allow("alice") == 0.0


# ---- fix tally ------------------------------------------------------------------------------------

@pytest.mark.parametrize("title, category", [
    ("Time-based stop instead of 10 steps", "Time budget, not a step limit"),
    ("Run until time budget, not 10 steps", "Time budget, not a step limit"),
    ("Show stderr and the tail of outputs", "Show stderr"),
    ("Show more output, keep head and tail", "Longer or smarter output"),
    ("Save full output to a searchable file", "Longer or smarter output"),
    ("Show the clock, force delivery near deadline", "Show the clock"),
    ("Verify requirements before declaring done", "Verify before finishing"),
    ("Detect repeated-command loops, force new approach", "Break loops and stalls"),
    ("Explicit work routine in system prompt", "Work routine in the prompt"),
    ("Add a per-command timeout", "Command timeouts"),
    ("Something entirely new", "Other"),
])
def test_fix_category_from_the_patch_title(title, category):
    assert evolve.fix_category(title, "") == category


def test_fix_category_falls_back_to_the_hypothesis():
    assert evolve.fix_category("Small tweak", "Hypothesis: stderr is discarded") == "Show stderr"


def test_diff_size_counts_changed_lines_not_headers():
    assert evolve.diff_size(DIFF) == (2, 1)
    assert evolve.diff_size("") == (0, 0)


# ---- board payload --------------------------------------------------------------------------------

def _events(*pairs):
    s = evolve.EvolveStore(clock=Clock())
    for user, e in pairs:
        s.add(user, evolve.validate(e))
    return s.events()


def test_state_curve_is_a_step_line_of_each_runs_current_harness():
    # best_val is the current harness's pooled validation score: dropped patches re-measure it (up or down).
    # The line changes level only at a kept patch; each level is that harness's latest score.
    ev = _events(("alice", seed(val=0.4)), ("alice", it(1, best=0.42)), ("alice", it(2, accepted=True, best=0.8)),
                 ("alice", it(3, best=0.78)), ("alice", it(4, best=0.79)))
    st = evolve.build_state(ev, reference=[])
    (run,) = st["runs"]
    assert run["curve"] == [[0, 0.42], [2, 0.42], [2, 0.79], [4, 0.79]]
    assert run["user"] == "alice" and run["kept"] == 1 and run["iterations"] == 4 and run["current"] == 0.79


def test_state_curve_of_a_seed_only_run_is_one_point():
    (run,) = evolve.build_state(_events(("alice", seed(val=0.4))), reference=[])["runs"]
    assert run["curve"] == [[0, 0.4]] and run["current"] == 0.4


def test_champion_curve_never_moves_between_kept_patches():
    ev = [seed(val=0.3)] + [it(i, accepted=i in (2, 5), best=0.3 + i / 20) for i in range(1, 8)]
    curve = evolve.champion_curve(ev)
    flat = [(a, b) for a, b in zip(curve, curve[1:]) if a[0] != b[0]]
    assert flat and all(a[1] == b[1] for a, b in flat)  # horizontal between keeps
    assert sorted({a[0] for a, b in zip(curve, curve[1:]) if a[0] == b[0]}) == [2, 5]  # vertical steps only at keeps


def test_state_totals_count_attendees_iterations_kept_and_latest_spend_per_run():
    ev = _events(("alice", seed(spend=0.1)), ("alice", it(1, accepted=True, best=0.8, spend=0.3)), ("alice", it(2, spend=0.5)),
                 ("alice", seed("alice-r2", spend=0.2)), ("bob", seed("bob-r1", spend=0.1)))
    t = evolve.build_state(ev, reference=[])["totals"]
    assert t == {"attendees": 2, "runs": 3, "iterations": 2, "kept": 1, "spend_usd": pytest.approx(0.8)}


def test_state_leader_is_the_highest_current_harness_at_each_iteration():
    ev = _events(("alice", seed(val=0.4)), ("alice", it(1, best=0.4)), ("alice", it(2, accepted=True, best=0.9)),
                 ("bob", seed("bob-r1", val=0.5)), ("bob", it(1, run_id="bob-r1", accepted=True, best=0.7)))
    st = evolve.build_state(ev, reference=[])
    assert st["leader"] == [[0, 0.5], [1, 0.5], [1, 0.7], [2, 0.7], [2, 0.9]]
    assert "room_best" not in st
    assert st["baseline"] == pytest.approx(0.45)  # mean seed validation score of all runs


def test_state_baseline_falls_back_to_the_recorded_runs_when_no_run_has_started():
    ref = [{"label": "recorded B1", "curve": [[0, 0.42], [1, 0.88]]}, {"label": "recorded B2", "curve": [[0, 0.33], [1, 0.83]]}]
    st = evolve.build_state([], reference=ref)
    assert st["baseline"] == pytest.approx(0.375)
    assert st["reference"] == ref
    assert st["runs"] == [] and st["totals"]["runs"] == 0


def test_feed_holds_kept_patches_newest_first_with_a_one_line_hypothesis_and_diff_size():
    hyp = "Hypothesis: runs end at the 10-step cap.\nSecond line that must not show."
    ev = _events(("alice", seed()), ("alice", it(1, accepted=True, best=0.8, hypothesis=hyp)), ("alice", it(2)),
                 ("bob", seed("bob-r1")), ("bob", it(1, run_id="bob-r1", accepted=True, best=0.7, title="Run until time budget")))
    feed = evolve.build_state(ev, reference=[])["feed"]
    assert [f["user"] for f in feed] == ["bob", "alice"]  # dropped iteration 2 is not in the feed
    a = feed[1]
    assert a["hypothesis"] == "runs end at the 10-step cap."  # no re-capitalising: it often starts with a task name
    assert (a["added"], a["removed"]) == (2, 1)
    assert a["delta"] == pytest.approx(0.4) and a["iteration"] == 1 and a["n_fixed"] == 1


def test_fixes_tally_counts_every_proposal_and_its_kept_share():
    ev = _events(("alice", seed()), ("alice", it(1, accepted=True, best=0.8, title="Time-based stop instead of 10 steps")),
                 ("alice", it(2, title="Show stderr to the model")),
                 ("bob", seed("bob-r1")), ("bob", it(1, run_id="bob-r1", accepted=True, best=0.7, title="Run until time budget, not 10 steps")))
    fixes = evolve.build_state(ev, reference=[])["fixes"]
    assert fixes[0] == {"label": "Time budget, not a step limit", "tried": 2, "kept": 2}
    assert fixes[1] == {"label": "Show stderr", "tried": 1, "kept": 0}


def test_state_lists_at_most_max_runs_best_first():
    pairs = []
    for i in range(evolve.MAX_RUNS + 5):
        pairs += [(f"u{i}", seed(f"u{i}-r", val=i / 1000))]
    st = evolve.build_state(_events(*pairs), reference=[])
    assert len(st["runs"]) == evolve.MAX_RUNS
    assert st["runs"][0]["current"] == pytest.approx((evolve.MAX_RUNS + 4) / 1000)
    assert st["totals"]["runs"] == evolve.MAX_RUNS + 5


def test_feed_hypothesis_is_the_whole_first_line_not_cut_mid_sentence():
    # the page fits it to two lines at a sentence or clause boundary; the server must not cut it first
    long = ("Hypothesis: all four transcripts end \"returned\" after exactly 10 model calls and only 18-32 s of the 180 s budget; "
            "the model was still exploring (listing files, checking tools) when the step cap cut it off.")
    ev = _events(("alice", seed()), ("alice", it(1, accepted=True, best=0.8, hypothesis=long)))
    (f,) = evolve.build_state(ev, reference=[])["feed"]
    assert f["hypothesis"] == long.removeprefix("Hypothesis: ") and "…" not in f["hypothesis"]
