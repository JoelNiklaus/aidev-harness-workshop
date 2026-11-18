"""Optimization strategies: the attendee-editable part of the loop (presets, knobs, the fixed fairness rules)."""

import pytest

from aidev_evolve import strategy


def test_four_presets_load_and_differ():
    names = strategy.presets()
    assert {"baseline", "failures-first", "small-diffs", "two-candidates"} <= set(names)
    loaded = [strategy.load(n) for n in names]
    assert len({s.hash for s in loaded}) == len(loaded)
    for s in loaded:
        assert s.name == s.preset and not s.modified
        assert s.prompt.strip(), f"{s.name} has an empty proposer prompt"
        assert s.description


def test_every_preset_fits_the_fixed_session_envelope():
    for n in strategy.presets():
        est = strategy.estimate(strategy.load(n))
        assert est["usd"] <= strategy.USD_CAP, (n, est)
        assert est["iterations_fit"] >= 2, (n, est)
        assert est["minutes"] <= strategy.SESSION_MINUTES + 1e-9, (n, est)


def test_presets_differ_in_the_knobs_they_advertise():
    b, f, s, t = (strategy.load(n) for n in ("baseline", "failures-first", "small-diffs", "two-candidates"))
    assert f.traces == "failed" and b.traces == "all"
    assert t.candidates == 2 and b.candidates == 1
    assert s.diffs is True and s.z < b.z
    assert "10 lines" in s.prompt or "ten lines" in s.prompt.lower()


def test_every_weakness_preset_fixes_each_mechanism_it_sees_in_one_patch():
    # The contrast to baseline's "one focused change": tally every failure mechanism in the failed transcripts and
    # fix each one seen in two or more of them, in the same patch (Phase 4: the second seed weakness is otherwise
    # left for an iteration whose keep margin it rarely clears).
    e, b = strategy.load("every-weakness"), strategy.load("baseline")
    assert e.traces == "failed" and e.candidates == 1 and e.max_traces == 8
    assert "one focused change" not in e.prompt and "one focused change" in b.prompt
    assert "every" in e.prompt.lower() and "two or more" in e.prompt.lower()
    assert "TIMED OUT" in e.prompt  # names the symptoms to look for, as limits-first did


def test_code_audit_preset_reads_the_harness_line_by_line_before_the_transcripts():
    # Phase 4: a weakness can leave no symptom in a transcript (a server the model checked, killed after its command),
    # so this preset starts from agent.py, not from failures, and fixes every harmful line in one patch.
    a = strategy.load("code-audit")
    assert a.traces == "all" and a.candidates == 1 and a.history == "full"
    assert "AUDIT.md" in a.prompt and "line by line" in a.prompt
    assert a.prompt.index("agent.py") < a.prompt.index("traces/")


def test_init_copies_a_preset_for_editing_and_the_copy_is_unmodified(tmp_path):
    d = strategy.init("small-diffs", tmp_path / "mine")
    assert (d / "strategy.toml").exists() and (d / "proposer_prompt.md").exists()
    s = strategy.load(str(d))
    assert s.preset == "small-diffs" and not s.modified and s.hash == strategy.load("small-diffs").hash
    with pytest.raises(FileExistsError):
        strategy.init("baseline", d)


def test_an_edited_prompt_or_knob_changes_the_hash_and_marks_it_modified(tmp_path):
    d = strategy.init("baseline", tmp_path / "mine")
    base = strategy.load(str(d))
    (d / "proposer_prompt.md").write_text(base.prompt + "\nPrefer changes to the system prompt.\n")
    edited = strategy.load(str(d / "strategy.toml"))  # a path to the toml works too
    assert edited.modified and edited.hash != base.hash and edited.preset == "baseline"
    toml = (d / "strategy.toml").read_text().replace("z = 1.5", "z = 1.0")
    (d / "strategy.toml").write_text(toml)
    assert strategy.load(str(d)).hash not in {base.hash, edited.hash}


@pytest.mark.parametrize("key", ["model", "budget_s", "heldout", "splits", "seed", "heldout_trials", "session_minutes"])
def test_fixed_settings_are_refused_for_fairness(tmp_path, key):
    d = strategy.init("baseline", tmp_path / "mine")
    (d / "strategy.toml").write_text((d / "strategy.toml").read_text() + f'\n[extra]\n{key} = 1\n')
    with pytest.raises(strategy.StrategyError, match=f"{key}.*fixed"):
        strategy.load(str(d))
    d2 = strategy.init("baseline", tmp_path / "top")
    (d2 / "strategy.toml").write_text(f"{key} = 1\n" + (d2 / "strategy.toml").read_text())
    with pytest.raises(strategy.StrategyError, match="fixed"):
        strategy.load(str(d2))


@pytest.mark.parametrize("old,new,msg", [
    ("candidates = 1", "candidates = 3", "candidates"),
    ("val_trials = 2", "val_trials = 0", "val_trials"),
    ("z = 1.5", "z = 9", "z"),
    ('traces = "all"', 'traces = "validation"', "traces"),
    ('history = "full"', 'history = "everything"', "history"),
    ("max_traces = 8", "max_traces = 0", "max_traces"),
    ("max_traces = 8", "max_traces = 2.5", "whole number"),
    ("confirm = true", "confirm = 1", "confirm"),
    ("iterations = 4", "iterations = 50", "iterations"),
    ("[keep]", "[keep]\nzz = 1", "unknown"),
])
def test_out_of_range_or_unknown_knobs_are_refused(tmp_path, old, new, msg):
    d = strategy.init("baseline", tmp_path / "mine")
    text = (d / "strategy.toml").read_text()
    assert old in text
    (d / "strategy.toml").write_text(text.replace(old, new))
    with pytest.raises(strategy.StrategyError, match=msg):
        strategy.load(str(d))


def test_a_strategy_over_the_credit_cap_is_refused(tmp_path):
    d = strategy.init("two-candidates", tmp_path / "mine")
    text = (d / "strategy.toml").read_text()
    text = text.replace("val_trials = 2", "val_trials = 3").replace("iterations = 3", "iterations = 6")
    (d / "strategy.toml").write_text(text)
    with pytest.raises(strategy.StrategyError, match=r"\$2"):
        strategy.load(str(d))


def test_unknown_preset_names_the_presets():
    with pytest.raises(strategy.StrategyError, match="baseline"):
        strategy.load("does-not-exist")


def test_the_event_block_carries_name_hash_knobs_and_the_prompt():
    s = strategy.load("failures-first")
    ev = s.event()
    assert ev["name"] == "failures-first" and ev["preset"] == "failures-first" and ev["hash"] == s.hash and ev["modified"] is False
    assert ev["knobs"] == s.knobs() and ev["knobs"]["traces"] == "failed"
    assert set(ev["knobs"]) == {"traces", "max_traces", "history", "diffs", "candidates", "val_trials", "z", "confirm", "iterations"}
    assert ev["prompt"] == s.prompt[:4000]


def test_the_time_box_starts_an_iteration_only_if_it_and_the_heldout_check_fit():
    s = strategy.load("baseline")
    session = 22 * 60
    # nothing measured yet: the prior iteration time is used
    assert strategy.next_fits(s, elapsed_s=3 * 60, iteration_s=[], session_s=session)
    # measured iterations of 4 min: at 12 min one more (4) + held-out fits, at 14 min it does not
    assert strategy.next_fits(s, elapsed_s=12 * 60, iteration_s=[240, 240], session_s=session)
    assert not strategy.next_fits(s, elapsed_s=14 * 60, iteration_s=[240, 240], session_s=session)
    # a slow proposer makes iterations longer, so fewer fit
    assert strategy.next_fits(s, elapsed_s=10 * 60, iteration_s=[240, 240], session_s=session)
    assert not strategy.next_fits(s, elapsed_s=10 * 60, iteration_s=[240, 420], session_s=session)
