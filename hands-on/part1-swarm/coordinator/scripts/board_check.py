# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.48"]
# ///
"""Board check in headless Chromium: load time (navigation start to the first full draw), screenshot at
projector size (1920x1080), and legibility: anything overflowing the viewport or its panel, clipped or
ellipsized text, the smallest font (HTML and the scatter's SVG text at screen scale; the projector minimum
is 26 px), and scatter text whose ink touches other text or a marker.

    uv run scripts/board_check.py --url http://127.0.0.1:8761/ --shot /tmp/board.png [--runs 3] [--header "Authorization: Bearer ..."]

Prints JSON. The page sets `data-drawn` on <html> after its first full render from /api/state.
The target is under 500 ms (engineering gate).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

from playwright.sync_api import sync_playwright


def chromium_path() -> str | None:
    found = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
    return found[-1] if found else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--shot")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--header", action="append", default=[], help="extra request header 'Name: value' (value never printed)")
    a = ap.parse_args()
    headers = dict(h.split(": ", 1) for h in a.header)
    out = {"url": a.url, "runs_ms": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path(), args=["--no-sandbox"])
        for i in range(a.runs):
            ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, extra_http_headers=headers)
            page = ctx.new_page()
            page.goto(a.url, wait_until="commit")
            page.wait_for_function("document.documentElement.dataset.drawn === '1'", timeout=15000)
            t = float(page.evaluate("document.documentElement.dataset.drawnAt"))
            nav = page.evaluate("JSON.stringify(performance.getEntriesByType('navigation')[0])")
            out["runs_ms"].append(round(t, 1))
            if i == a.runs - 1:
                n = json.loads(nav)
                out["html_ms"] = round(n.get("responseEnd", 0), 1)
                out["html_bytes"] = n.get("transferSize")
                out["overflow"] = page.evaluate("""(() => {
                    const d = document.documentElement;
                    const bad = [...document.querySelectorAll('td,th,.gname,.feed li,h1,h2,.plegend,.legend,.nr,.counter')].filter(e => e.scrollWidth > e.clientWidth + 1 && getComputedStyle(e).textOverflow !== 'ellipsis').length;
                    const cut = [...document.querySelectorAll('tbody tr,.gname,.legend,.feed li')].filter(e => { const p = e.closest('.panel'); return p && e.getBoundingClientRect().bottom > p.getBoundingClientRect().bottom - 4; }).length;
                    return {page_w: d.scrollWidth, page_h: d.scrollHeight, clipped_text_nodes: bad, rows_cut_by_panel: cut};
                })()""")
                out["min_font_px"] = page.evaluate("""Math.min(...[...document.querySelectorAll('body *')].filter(e => e.offsetParent && e.childNodes.length && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())).map(e => parseFloat(getComputedStyle(e).fontSize)))""")
                out["scatter"] = page.evaluate("""(() => {
                    // SVG text at screen scale, and the ink of texts that touches other ink or a marker
                    // (font boxes include line gap above and below the glyphs, so they are measured on canvas)
                    const svg = document.getElementById('pareto'), k = svg.getScreenCTM().a, ctx = document.createElement('canvas').getContext('2d');
                    const texts = [...svg.querySelectorAll('text')].map(t => {
                        const r = t.getBoundingClientRect(), cs = getComputedStyle(t);
                        ctx.font = cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
                        const m = ctx.measureText(t.textContent), base = svg.getBoundingClientRect().top + parseFloat(t.getAttribute('y')) * k;
                        return {t: t.textContent, px: parseFloat(t.getAttribute('font-size')) * k,
                                b: {left: r.left, right: r.right, top: base - m.actualBoundingBoxAscent, bottom: base + m.actualBoundingBoxDescent}};
                    });
                    const marks = [...svg.querySelectorAll('circle')].map(c => c.getBoundingClientRect());
                    const hit = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
                    const pairs = [];
                    texts.forEach((a, i) => texts.slice(i + 1).forEach(b => { if (hit(a.b, b.b)) pairs.push(a.t + ' / ' + b.t); }));
                    const onMarker = texts.filter(a => marks.some(m => hit(a.b, m))).map(a => a.t);
                    return {min_font_px: Math.min(...texts.map(a => a.px)), text_overlaps: pairs, text_on_marker: onMarker, n_text: texts.length};
                })()""")
                out["ellipsized"] = page.evaluate("""[...document.querySelectorAll('body *')].filter(e => e.offsetParent && e.scrollWidth > e.clientWidth + 1 && getComputedStyle(e).textOverflow === 'ellipsis').map(e => e.textContent.trim())""")
                if a.shot:
                    Path(a.shot).parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=a.shot)
                    out["shot"] = a.shot
            ctx.close()
        browser.close()
    out["median_ms"] = statistics.median(out["runs_ms"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
