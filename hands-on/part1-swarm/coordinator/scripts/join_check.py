# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.48"]
# ///
"""Measure the /join page in headless Chromium on a phone and a laptop viewport.

    uv run scripts/join_check.py --url http://127.0.0.1:7860/join --shot /tmp/join.png [--header "Authorization: Bearer ***"]

Per viewport: load time (navigation start to load event, median of --runs), smallest font size, horizontal
overflow (a box wider than the screen), and whether each Copy button puts its box's exact text on the clipboard.
"""

from __future__ import annotations

import argparse
import json
import statistics

from playwright.sync_api import sync_playwright

from board_check import chromium_path

VIEWPORTS = {"phone": {"width": 390, "height": 844}, "laptop": {"width": 1440, "height": 900}}

PROBE = """() => {
  const t = performance.getEntriesByType('navigation')[0];
  const sizes = [...document.querySelectorAll('body *')].filter(e => e.childNodes.length && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()))
    .map(e => parseFloat(getComputedStyle(e).fontSize));
  return {load_ms: Math.round(t.loadEventEnd - t.startTime), min_font_px: Math.min(...sizes),
          overflow_px: Math.max(0, document.documentElement.scrollWidth - innerWidth)};
}"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--shot", help="phone and laptop screenshots: <shot> and <shot stem>-phone.png")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--header", action="append", default=[], help='"Name: value", e.g. the Space token')
    a = ap.parse_args(argv)
    headers = dict(h.split(": ", 1) for h in a.header)
    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path())
        for name, vp in VIEWPORTS.items():
            ctx = browser.new_context(viewport=vp, extra_http_headers=headers, permissions=["clipboard-read", "clipboard-write"])
            page = ctx.new_page()
            runs = []
            for _ in range(a.runs):
                page.goto(a.url, wait_until="load")
                runs.append(page.evaluate(PROBE))
            copies = {}
            for b in page.query_selector_all("button[data-copy]"):
                box = b.get_attribute("data-copy")
                b.click()
                copies[box] = page.evaluate("navigator.clipboard.readText()") == page.inner_text(f"#{box}")
            out[name] = {"load_ms_median": statistics.median(r["load_ms"] for r in runs), "load_ms": [r["load_ms"] for r in runs],
                         "min_font_px": runs[-1]["min_font_px"], "overflow_px": runs[-1]["overflow_px"], "copy_ok": copies}
            if a.shot:
                page.goto(a.url, wait_until="load")
                shot = a.shot if name == "laptop" else a.shot.removesuffix(".png") + "-phone.png"
                page.screenshot(path=shot, full_page=True)
            ctx.close()
        browser.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
