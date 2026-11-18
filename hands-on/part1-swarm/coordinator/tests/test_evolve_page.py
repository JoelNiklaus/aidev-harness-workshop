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
    for heading in ("Race · best validation pass rate by iteration", "Most common fixes", "Newest kept patches"):
        assert heading in PAGE
    for label in ("attendees", "iterations", "patches kept", "HF credits"):
        assert f'<div class="lab">{label}' in PAGE
    for key in ("room best", "seed baseline"):
        assert key in PAGE  # race legend, under the plot


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
