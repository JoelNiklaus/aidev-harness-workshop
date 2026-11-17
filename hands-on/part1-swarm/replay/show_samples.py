"""Render three diverse reference trials as HTML cards (data check for the replay/reference layer).

    python3 replay/show_samples.py [--out <file.html>]

Picks: the cheapest pass, the most expensive fail, and the longest trial of a third harness;
each card says why it was picked.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).parent


def pick(results: list[dict]) -> list[tuple[str, dict]]:
    passes = [r for r in results if r["passed"]]
    fails = [r for r in results if not r["passed"]]
    a = min(passes, key=lambda r: r["cost_usd"])
    b = max(fails, key=lambda r: r["cost_usd"])
    rest = [r for r in results if r["harness"] not in (a["harness"], b["harness"])]
    c = max(rest, key=lambda r: r["wall_s"])
    return [("cheapest pass", a), ("most expensive fail", b), ("longest trial, a third harness", c)]


def card(why: str, r: dict) -> str:
    rows = [("cell", r["cell_id"]), ("harness", f'{r["harness"]} {r["harness_version"]}'), ("task", r["task"]),
            ("outcome", "pass" if r["passed"] else "fail"), ("reward", r["reward"]), ("wall time", f'{r["wall_s"]/60:.1f} min'),
            ("model calls", r["n_calls"]), ("input tokens", f'{r["n_input_tokens"]:,} ({(r["n_cached_tokens"] or 0):,} cached)'),
            ("output tokens", f'{r["n_output_tokens"]:,}'), ("cost", f'${r["cost_usd"]:.4f}'),
            ("source", f'{r["source_run"]} / {r["source_trial_id"]}')]
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f'<section class="card"><h2>{html.escape(why)}</h2><table>{body}</table></section>'


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(HERE / "reference.json"))
    ap.add_argument("--out", default=str(HERE / "reference-samples.html"))
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.ref).read_text())
    cards = "".join(card(w, r) for w, r in pick(doc["results"]))
    page = f"""<!doctype html><meta charset="utf-8"><title>Reference trials: three samples</title>
<style>body{{font:16px/1.4 system-ui,sans-serif;margin:20px;max-width:900px}}.card{{border:1px solid #ccc;border-radius:10px;padding:12px 16px;margin:14px 0}}
h2{{margin:0 0 8px;font-size:18px}}table{{border-collapse:collapse;width:100%}}th{{text-align:left;width:140px;color:#555;font-weight:600;vertical-align:top;padding:3px 8px 3px 0}}
td{{padding:3px 0;overflow-wrap:anywhere}}</style>
<h1>Reference layer: three of {doc['n']} trials</h1>
<p>{html.escape(doc['source'])}; {html.escape(doc['model'])}, effort {html.escape(doc['effort'])}, {html.escape(doc['benchmark'])}; snapshot {html.escape(str(doc['snapshot_loaded_at']))}.</p>{cards}"""
    Path(a.out).write_text(page)
    print(a.out)


if __name__ == "__main__":
    main()
