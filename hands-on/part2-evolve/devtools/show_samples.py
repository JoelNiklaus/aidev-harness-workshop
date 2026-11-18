"""Render three diverse trials of an evolve run as readable HTML cards (engineering gate: look at the data).

    python devtools/show_samples.py runs/phase1-real "<talk folder>/evolve/phase1-samples.html"

Picks, for spread: a seed failure that hit the step limit, a seed failure that stopped early (few model
calls), and a pass of the best agent. Each card shows the digest the proposer reads (digest.render).
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidev_evolve import digest  # noqa: E402


def pick(records: list[dict], best: str) -> list[tuple[str, dict]]:
    calls = lambda r: ((r.get("trajectory") or {}).get("usage") or {}).get("calls", 0)  # noqa: E731
    seed_fail = [r for r in records if r["candidate"] == "c000" and r["outcome"] == "fail" and r.get("trajectory")]
    out = []
    long = max(seed_fail, key=calls, default=None)
    if long:
        out.append(("Seed failure: used every step", long))
    short = min((r for r in seed_fail if r is not long), key=calls, default=None)
    if short:
        out.append(("Seed failure: stopped early", short))
    passes = [r for r in records if r["candidate"] == best and r["outcome"] == "pass" and r.get("trajectory")]
    if passes:
        out.append((f"Pass by the best agent ({best})", max(passes, key=calls)))
    return out


def main(run_dir: str, out: str) -> None:
    run = Path(run_dir)
    records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines() if line.strip()]
    best = json.loads((run / "state.json").read_text())["incumbent"]
    cards = []
    for why, r in pick(records, best):
        t = r["trajectory"]
        meta = f"task {r['task']} · split {r['split']} · {r['outcome'].upper()} · ended {t['end']} · {t['usage']['calls']} model calls · {r.get('wall_s') or 0:.0f} s · ${r['cost_usd']:.4f}"
        cards.append(f"<section><h2>{html.escape(why)}</h2><p class=meta>{html.escape(meta)}</p><pre>{html.escape(digest.render(r, max_chars=9000))}</pre></section>")
    page = ("<!doctype html><meta charset=utf-8><title>evolve samples</title><style>body{font:15px system-ui;max-width:900px;margin:20px auto;padding:0 12px}"
            "section{border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0}.meta{color:#555}pre{white-space:pre-wrap;word-break:break-word;"
            "background:#f6f6f6;padding:10px;max-height:600px;overflow:auto}</style>"
            f"<h1>Three trials from {html.escape(run.name)}</h1>" + "".join(cards))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(page)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
