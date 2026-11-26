"""chart.html: score by version, as one self-contained page.

Plain HTML with an inline SVG drawn here: no script, no external asset, no server. It opens by double-click and
reloads itself every 5 seconds (`eval` rewrites the file), so the newest version appears without a click.
"""

from __future__ import annotations

from html import escape
from typing import Any

WIDTH, LEFT, RIGHT = 1200, 90, 40
TOP, PLOT_H = 100, 310  # the y of 100% and the height of the plot
BAR_MAX_W = 170
HELD_ROW_H = 58

CSS = """
body { margin: 0; background: #fff; color: #111; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 12px 20px; }
h1 { font-size: 34px; margin: 8px 0 0; }
p { font-size: 20px; color: #444; margin: 4px 0 8px; }
svg { width: 100%; height: auto; max-height: calc(100vh - 120px); display: block; }  /* the whole chart fits the window: no scrolling */
.bar { fill: #1565c0; } .held { fill: #2e7d32; }
.pct { font-size: 46px; font-weight: 700; text-anchor: middle; }
.count { font-size: 26px; fill: #333; text-anchor: middle; }
.name { font-size: 40px; font-weight: 700; text-anchor: middle; }
.axis { font-size: 20px; fill: #777; text-anchor: end; }
.grid { stroke: #ddd; stroke-width: 1; } .base { stroke: #333; stroke-width: 2; }
.htitle { font-size: 30px; font-weight: 700; }
.many .pct { font-size: 28px; } .many .count { font-size: 18px; } .many .name { font-size: 28px; }
.hname { font-size: 32px; font-weight: 700; } .hval { font-size: 32px; }
p.empty { font-size: 26px; color: #111; margin: 16px 0; }
"""


EMPTY = "No version yet: run `aidev-evolve eval` in this folder and version 1 appears here."


def _pct(passed: int, scored: int) -> str:
    return f"{100 * passed / scored:.0f}%" if scored else "-"


def render(versions: list[dict[str, Any]], *, updated: str = "") -> str:
    """The page for these versions (workspace.py's entries: version, passed, scored, optional heldout); none yet: one sentence."""
    n = max(1, len(versions))
    slot = (WIDTH - LEFT - RIGHT) / n
    bar_w = min(BAR_MAX_W, slot * 0.62)
    base_y = TOP + PLOT_H
    out = []
    for frac in (0, 0.25, 0.5, 0.75, 1):
        y = base_y - frac * PLOT_H
        out.append(f'<line class="{"base" if frac == 0 else "grid"}" x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}"/>'
                   f'<text class="axis" x="{LEFT - 10}" y="{y + 7:.1f}">{100 * frac:.0f}%</text>')
    for i, v in enumerate(versions):
        cx = LEFT + slot * (i + 0.5)
        h = PLOT_H * (v["passed"] / v["scored"] if v["scored"] else 0)
        out.append(f'<rect class="bar" x="{cx - bar_w / 2:.1f}" y="{base_y - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}"/>')
        out.append(f'<text class="pct" x="{cx:.1f}" y="{base_y - h - 44:.1f}">{_pct(v["passed"], v["scored"])}</text>')
        out.append(f'<text class="count" x="{cx:.1f}" y="{base_y - h - 12:.1f}">{v["passed"]}/{v["scored"]}</text>')
        out.append(f'<text class="name" x="{cx:.1f}" y="{base_y + 46}">v{v["version"]}</text>')
    height = base_y + 90
    held = [v for v in versions if v.get("heldout")]
    if held:
        out.append(f'<text class="htitle" x="{LEFT}" y="{height + 34}">Held-out tasks (never read while fixing)</text>')
        height += 46
        for v in held:
            h = v["heldout"]
            w = (WIDTH - LEFT - RIGHT - 420) * (h["passed"] / h["scored"] if h["scored"] else 0)
            label = f'{_pct(h["passed"], h["scored"])} ({h["passed"]}/{h["scored"]})'
            out.append(f'<text class="hname" x="{LEFT}" y="{height + 40}">v{v["version"]}</text>'
                       f'<rect class="held" x="{LEFT + 70}" y="{height + 8}" width="{w:.1f}" height="44"/>'
                       f'<text class="hval" x="{LEFT + 84 + w:.1f}" y="{height + 41}">{escape(label)}</text>')
            height += HELD_ROW_H
    empty = "" if versions else f'\n<p class="empty">{escape(EMPTY)}</p>'
    note = "8 validation tasks, 1 trial each. This page reloads itself every 5 s" + (f"; written {escape(updated)}" if updated else "") + "."
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>aidev-evolve: score by version</title><style>{CSS}</style></head>\n'
            f'<body><main><h1>Score by version of agent.py</h1><p>{note}</p>{empty}\n'
            f'<svg{' class="many"' if n > 7 else ""} viewBox="0 0 {WIDTH} {height:.0f}" role="img" aria-label="Pass rate by version">\n' + "\n".join(out) + "\n</svg></main></body></html>\n")
