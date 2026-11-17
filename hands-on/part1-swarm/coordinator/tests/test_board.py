"""The board page (static/board.html) as the projector shows it: text sizes for a 250-500-person room,
leaderboard columns, and the scatter's label placement (run in Node against the page's own function)."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from importlib import resources

import pytest

BOARD = resources.files("swarm_coordinator").joinpath("static/board.html").read_text()
MIN_PX = 26  # back row ~30 m from a 1080p projector


def test_no_text_smaller_than_the_projector_minimum():
    sizes = [int(v) for v in re.findall(r"font-size:\s*(\d+)px", BOARD)]  # CSS rules and inline styles
    sizes += [int(v) for v in re.findall(r"""font-size=\\?["'](\d+)""", BOARD)]  # SVG text built in the script
    sizes += [int(v) for v in re.findall(r"font:\s*\d+\s+(\d+)px", BOARD)]  # font shorthands
    assert sizes, "no font sizes found"
    small = sorted(s for s in sizes if s < MIN_PX)
    assert not small, f"font sizes below {MIN_PX}px: {small}"


def test_leaderboard_shows_the_story_columns_only():
    # the story's two axes (pass rate, cost per trial) plus trials and the reference; $/pass and tokens/trial
    # were dropped so every header fits on one line at projector size
    head = re.search(r"<thead>(.*?)</thead>", BOARD, re.S).group(1)
    cols = [" ".join(re.sub(r"<[^>]+>", " ", c).split()) for c in re.findall(r"<th[^>]*>(.*?)</th>", head, re.S)]
    assert cols == ["Harness", "Pass rate", "95% CI", "Ref", "Trials", "$ / trial"]


def test_scatter_legend_names_room_and_reference_markers():
    # the 95% CI lives in the leaderboard's bars; whiskers in the scatter fenced the labels in
    legend = re.search(r'<div class="plegend"[^>]*>(.*?)</div>', BOARD, re.S)
    assert legend, "the scatter needs a legend outside the plot"
    text = re.sub(r"<[^>]+>", " ", legend.group(1))
    assert "this room" in text and "Harness Leaderboard" in text


# ---- label placement -------------------------------------------------------------------------------

def _place_labels_source() -> str:
    m = re.search(r"// <placeLabels>(.*?)// </placeLabels>", BOARD, re.S)
    assert m, "board.html lacks the placeLabels block"
    return m.group(1)


def _run_placement(items, obstacles, bounds):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed: the label placement runs in the page's JavaScript")
    prog = _place_labels_source() + "\nconst a = JSON.parse(process.argv[1]);\n" \
        "process.stdout.write(JSON.stringify(placeLabels(a.items, a.obstacles, a.bounds)));\n"
    out = subprocess.run([node, "-e", prog, json.dumps({"items": items, "obstacles": obstacles, "bounds": bounds})],
                         capture_output=True, text=True, timeout=20, check=True)
    return json.loads(out.stdout)


def _overlap(a, b, pad=0.0):
    return a[0] < b[2] + pad and a[2] > b[0] - pad and a[1] < b[3] + pad and a[3] > b[1] - pad


def _box(p):
    return (p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])


def _circle_hits_box(cx, cy, r, b):
    dx = max(b[0] - cx, 0, cx - b[2])
    dy = max(b[1] - cy, 0, cy - b[3])
    return dx * dx + dy * dy < r * r


def _segment_hits_box(seg, b, step=1.0):
    x1, y1, x2, y2 = seg
    n = max(1, int(math.hypot(x2 - x1, y2 - y1) / step))
    return any(b[0] < x1 + (x2 - x1) * i / n < b[2] and b[1] < y1 + (y2 - y1) * i / n < b[3] for i in range(n + 1))


def _segments_cross(a, b):
    def side(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 1e-9) - (v < -1e-9)
    a1, a2, b1, b2 = a[:2], a[2:], b[:2], b[2:]
    return side(a1, a2, b1) * side(a1, a2, b2) < 0 and side(b1, b2, a1) * side(b1, b2, a2) < 0


def _label(name, x, y, r=15, px_per_char=16.0, h=32):
    return {"id": name, "x": x, "y": y, "r": r, "w": round(len(name) * px_per_char + 4), "h": h}


def _check_layout(items, obstacles, bounds, placed):
    """Legibility invariants, computed here independently of the page's code."""
    assert [p["id"] for p in placed] == [i["id"] for i in items], "every label placed, in input order"
    boxes = {p["id"]: _box(p) for p in placed}
    for p in placed:
        b = boxes[p["id"]]
        assert bounds["x0"] <= b[0] and b[2] <= bounds["x1"] and bounds["y0"] <= b[1] and b[3] <= bounds["y1"], (p, bounds)
        for q in placed:
            if q["id"] != p["id"]:
                assert not _overlap(b, boxes[q["id"]]), f"labels overlap: {p['id']} / {q['id']}"
        for it in items:  # no label on any marker, its own included
            assert not _circle_hits_box(it["x"], it["y"], it["r"], b), f"label {p['id']} covers marker {it['id']}"
        for o in obstacles:
            if "seg" in o:
                assert not _segment_hits_box(o["seg"], b), f"label {p['id']} crosses a line {o}"
            else:
                assert not _overlap(b, (o["x0"], o["y0"], o["x1"], o["y1"])), f"label {p['id']} covers {o}"
        it = next(i for i in items if i["id"] == p["id"])
        dx, dy = max(b[0] - it["x"], 0, it["x"] - b[2]), max(b[1] - it["y"], 0, it["y"] - b[3])
        if p["leader"] is None:
            assert math.hypot(dx, dy) <= it["r"] + 12, f"label {p['id']} drifted from its point without a leader line"
        else:
            seg = (p["leader"]["x1"], p["leader"]["y1"], p["leader"]["x2"], p["leader"]["y2"])
            for q in placed:
                if q["id"] != p["id"]:
                    assert not _segment_hits_box(seg, boxes[q["id"]]), f"leader of {p['id']} crosses label {q['id']}"
            lines = [o["seg"] for o in obstacles if "seg" in o]
            lines += [[q["leader"][k] for k in ("x1", "y1", "x2", "y2")] for q in placed if q["leader"] and q["id"] != p["id"]]
            for ln in lines:
                assert not _segments_cross(seg, ln), f"leader of {p['id']} crosses a line {ln}"


BOUNDS = {"x0": 90, "y0": 0, "x1": 780, "y1": 330}


def test_crowded_cluster_from_the_midrun_board_gets_separate_labels():
    # the overlap on the 2026-11-17 mid-run render: Kimi Code, Claude Code and mini-swe-agent within ~40 px,
    # each next to its hollow reference marker and with a CI whisker through it
    pts = [("Kimi Code", 560, 120), ("Claude Code", 575, 128), ("mini-swe-agent", 548, 112), ("Pi", 300, 128),
           ("OpenCode", 285, 125), ("DeepSeek Harness", 520, 168), ("OpenClaw", 690, 240), ("Hermes Agent", 400, 150),
           ("Codex", 455, 170), ("Qwen Code", 640, 210)]
    items = [_label(n, x, y) for n, x, y in pts]
    obstacles = []
    for n, x, y in pts:
        obstacles.append({"x0": x + 25 - 14, "y0": y + 4 - 14, "x1": x + 25 + 14, "y1": y + 4 + 14})  # reference ring
        obstacles.append({"seg": [x, y - 45, x, y + 45]})  # CI whisker
    placed = _run_placement(items, obstacles, BOUNDS)
    _check_layout(items, obstacles, BOUNDS, placed)


def test_ten_coincident_points_still_get_readable_labels_with_leader_lines():
    items = [_label(n, 430 + i % 3, 160 + i % 2) for i, n in enumerate(
        ["Kimi Code", "Claude Code", "mini-swe-agent", "Pi", "OpenCode", "DeepSeek Harness", "OpenClaw",
         "Hermes Agent", "Codex", "Qwen Code"])]
    placed = _run_placement(items, [], BOUNDS)
    _check_layout(items, [], BOUNDS, placed)
    assert sum(p["leader"] is not None for p in placed) >= 5


def test_isolated_point_label_sits_next_to_it_without_a_leader():
    items = [_label("OpenClaw", 400, 160)]
    placed = _run_placement(items, [], BOUNDS)
    _check_layout(items, [], BOUNDS, placed)
    assert placed[0]["leader"] is None and placed[0]["x"] > 400  # right of the point, the reading direction


def test_point_at_the_right_edge_puts_its_label_inside_the_plot():
    items = [_label("DeepSeek Harness", 770, 10)]
    placed = _run_placement(items, [], BOUNDS)
    _check_layout(items, [], BOUNDS, placed)
