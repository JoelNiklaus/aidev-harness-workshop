"""The /evolve projector page (static/evolve.html): text sizes for a 250-500-person room, the race chart's
legend, and the end-of-line label stacking (run in Node against the page's own function)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from importlib import resources

import pytest

PAGE = resources.files("swarm_coordinator").joinpath("static/evolve.html").read_text()
MIN_PX = 26


def test_no_text_smaller_than_the_projector_minimum():
    sizes = [int(v) for v in re.findall(r"font-size:\s*(\d+)px", PAGE)]
    sizes += [int(v) for v in re.findall(r"font:\s*\d+\s+(\d+)px", PAGE)]
    sizes += [int(v) for v in re.findall(r"const FS = (\d+);", PAGE)]  # every SVG text uses FS
    assert sizes and min(sizes) >= MIN_PX, sorted(sizes)
    assert not re.search(r'font-size="\d', PAGE), "SVG text must use FS, not a literal size"


def test_page_has_the_four_parts_the_brief_asks_for():
    assert "<h1>Evolve the harness together<" in PAGE and "<title>Evolve the harness together</title>" in PAGE
    for heading in ("Validation pass rate of each run's current harness", "Most common fixes", "Newest kept patches"):
        assert heading in PAGE
    for label in ("attendees", "iterations", "patches kept", "HF credits"):
        assert f'<div class="lab">{label}' in PAGE
    for key in ("leading run", "seed baseline"):
        assert key in PAGE  # race legend, under the plot


def test_page_never_names_the_room_and_never_calls_a_line_best():
    # Joel: "the room" is not an actor or a label; a line that can step down must not be called "best"
    shown = re.sub(r"//[^\n]*|/\*.*?\*/", "", PAGE, flags=re.S)
    assert not re.search(r"\bthe room\b|room best", shown, re.I)
    assert not re.search(r"best validation", shown, re.I)
    assert "s.room_best" not in PAGE and "s.leader" in PAGE


def test_untrusted_text_goes_through_esc():
    # every user-controlled field the page renders: username, title, hypothesis, fix label
    for field in ("f.user", "f.title", "f.hypothesis", "f.label", "name"):
        assert f"${{esc({field})}}" in PAGE, field


def _stack(ys, gap, lo, hi):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed: the label stacking runs in the page's JavaScript")
    src = re.search(r"// <stackLabels>(.*?)// </stackLabels>", PAGE, re.S).group(1)
    prog = src + "\nconst a = JSON.parse(process.argv[1]);\nprocess.stdout.write(JSON.stringify(stackLabels(a.ys, a.gap, a.lo, a.hi)));\n"
    out = subprocess.run([node, "-e", prog, json.dumps({"ys": ys, "gap": gap, "lo": lo, "hi": hi})], capture_output=True, text=True, timeout=20, check=True)
    return json.loads(out.stdout)


def test_labels_far_apart_stay_on_their_lines():
    assert _stack([100, 300, 200], 32, 0, 600) == [100, 300, 200]


def test_coincident_labels_are_spread_by_the_gap_and_keep_their_order():
    out = _stack([200, 200, 200, 210, 500], 32, 0, 600)
    s = sorted(out[:4])
    assert all(b - a >= 32 for a, b in zip(s, s[1:]))
    assert out[4] == 500


def test_labels_piled_at_the_bottom_are_pushed_up_inside_the_bounds():
    out = _stack([590, 595, 600, 600, 600], 32, 13, 590)
    s = sorted(out)
    assert s[-1] <= 590 and s[0] >= 13
    assert all(b - a >= 32 for a, b in zip(s, s[1:]))


# ---- patch feed: the hypothesis fits two lines and ends at a sentence or clause boundary --------------

def _cands(text):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed: the hypothesis cut runs in the page's JavaScript")
    src = re.search(r"// <hypCandidates>(.*?)// </hypCandidates>", PAGE, re.S).group(1)
    prog = src + "\nprocess.stdout.write(JSON.stringify(hypCandidates(process.argv[1])));\n"
    out = subprocess.run([node, "-e", prog, text], capture_output=True, text=True, timeout=20, check=True)
    return json.loads(out.stdout)


B1 = ("db-wal-recovery, financial-document-processor and sparql-university ended at the 10-step cap after only "
      "18-50 s of 180 s, still mid-work.")
B2 = ('all four transcripts end "returned" after exactly 10 model calls and only 18-32 s of the 180 s budget; '
      "the model was still exploring (listing files, checking tools) when the step cap cut it off.")


def test_hypothesis_candidates_end_at_a_boundary_longest_first():
    for text in (B1, B2):
        c = _cands(text)
        assert c[0] == text and c == sorted(c, key=len, reverse=True)
        for s in c:
            assert s[-1] in ".!?" and "…" not in s and len(s.split()) >= 4, s
            assert text.startswith(s[:-1].rstrip()), s  # a prefix of the hypothesis, closed with a full stop
            assert s.count("(") == s.count(")"), s


def test_hypothesis_candidates_cut_at_semicolons_and_conjunctions_not_inside_a_list():
    c = _cands(B2)
    assert 'all four transcripts end "returned" after exactly 10 model calls and only 18-32 s of the 180 s budget.' in c
    assert 'all four transcripts end "returned" after exactly 10 model calls.' in c
    assert all(not s.startswith("db-wal-recovery.") and s != "db-wal-recovery, financial-document-processor." for s in _cands(B1))
    assert B1 in _cands(B1)
    # before a subordinate word (after, when, until, ...): the main clause alone is still a sentence
    assert "db-wal-recovery, financial-document-processor and sparql-university ended at the 10-step cap." in _cands(B1)


def test_hypothesis_without_a_boundary_has_only_itself():
    assert _cands("runs end at the step cap") == ["runs end at the step cap"]


# ---- strategies (part2-evolve/REQUESTS.md: Strategies panel and Winner view) ----------------------------


def test_page_has_a_strategies_panel_by_preset_first_then_by_exact_strategy():
    assert ">Strategies<" in PAGE
    assert PAGE.index("by preset") < PAGE.index("edited copies")
    assert "s.presets" in PAGE and "s.strategies" in PAGE
    assert "needs ${" in PAGE and "s.min_runs" in PAGE  # unranked rows say how many runs they need
    assert "held-out gain" in PAGE.lower() and "95% CI" in PAGE


def test_strategy_rows_show_shared_ranks_with_an_equals_sign_and_bracket_tie_groups():
    assert '"=" +' in PAGE or "'=' +" in PAGE
    for cls in ("tie-top", "tie-mid", "tie-bot"):
        assert cls in PAGE


def test_winner_view_names_the_strategy_or_the_tie_and_shows_the_winning_patch():
    for text in ("Winning strategy", "Tie for first", "Winning patch", "one run: about ±15-20 points", "view=winner"):
        assert text in PAGE, text
    assert "diff_url" in PAGE and "knobs" in PAGE and "prompt" in PAGE


def test_strategy_and_patch_text_goes_through_esc():
    # strategy names, presets, prompts, kept titles and diffs come from attendees
    for field in ("e.name", "e.preset", "st.name", "t", "line", "p.user", "p.run_id"):
        assert f"${{esc({field})}}" in PAGE, field
    assert "${esc(st.prompt." in PAGE
