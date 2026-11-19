"""Build replay/evolve-recorded.json: recorded evolve runs as the events their loops would have posted to /evolve
(seed, one per iteration, held-out), with the time each was printed.

    python3 replay/build_evolve_replay.py --runs <talk folder>/evolve/runs/p4/loops      # every run folder in it

Reads each run folder's `state.json` (history, curve, held-out, spend), `config.json` (the optimization strategy,
when the run had one) and `transcript.jsonl` (when the seed summary, each iteration card and the end were printed).
The events mirror `aidev_evolve.loop` (`_submit`, client 0.5.0): the strategy block and `provider` (config.json's first
provider, when the run had one) on every event, the iteration's `hypothesis`, `siblings` and `provider_calls`, the held-out
`per_task` counts, `kept` titles and `provider_calls`. Stdlib only; the coordinator's tests
validate the output.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = "aidev-evolve/1"


def times(transcript: Path) -> tuple[float, dict[int, float], float]:
    seed_t, its, end = None, {}, 0.0
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        t, text = float(row["t"]), row["text"]
        end = max(end, t)
        if seed_t is None and text.lstrip().startswith("Seed agent:"):
            seed_t = t
        m = re.match(r"\s*Iteration (\d+) ", text)
        if m:
            its.setdefault(int(m.group(1)), t)
    if seed_t is None:
        raise SystemExit(f"{transcript}: no seed summary line")
    return seed_t, its, end


def search_score(transcript: Path) -> float | None:
    m = re.search(r"Seed agent: search\s+(\d+)%", transcript.read_text())
    return int(m.group(1)) / 100 if m else None


def strategy_block(folder: Path) -> dict | None:
    """The `strategy` field the client sends (aidev_evolve.strategy.Strategy.event), from the run's config.json."""
    cfg = folder / "config.json"
    st = json.loads(cfg.read_text()).get("strategy") if cfg.exists() else None
    if not st:
        return None
    return {"name": st["name"], "preset": st["preset"], "hash": st["hash"], "modified": st["modified"], "knobs": st["knobs"],
            "prompt": st["prompt"][:4000]}


def run_events(folder: Path) -> list[dict]:
    state = json.loads((folder / "state.json").read_text())
    strategy = strategy_block(folder)
    seed_t, its, end = times(folder / "transcript.jsonl")
    hist = state["history"]
    seed_spend = round(hist[0]["spend_usd"] - hist[0]["iter_usd"], 6) if hist else state["spend_usd"]
    rid = folder.name
    events = [{"t": seed_t, "schema": SCHEMA, "run_id": rid, "kind": "seed", "val_score": state["curve"][0],
               "search_score": search_score(folder / "transcript.jsonl"), "spend_usd": seed_spend}]
    for h in hist:
        events.append({"t": its[h["iteration"]], "schema": SCHEMA, "run_id": rid, "kind": "iteration", "iteration": h["iteration"],
                       "candidate": h["candidate"], "parent": h["parent"], "title": h.get("title"), "hypothesis": h.get("hypothesis"),
                       "accepted": h["verdict"] == "kept", "verdict": h["verdict"], "val_child": h.get("val_child"),
                       "val_parent": h.get("val_parent"), "best_val": h["best_val"], "fixed": h.get("fixed", []),
                       "broke": h.get("broke", []), "diff": (h.get("diff") or "")[:8000], "spend_usd": h["spend_usd"],
                       "siblings": h.get("siblings", [])})
        if h.get("provider_calls"):
            events[-1]["provider_calls"] = h["provider_calls"]
    ho = state.get("heldout")
    if ho:
        events.append({"t": end, "schema": SCHEMA, "run_id": rid, "kind": "heldout",
                       **{k: ho.get(k) for k in ("seed", "best", "best_candidate", "n_tasks", "trials", "diff", "ci_low", "ci_high")},
                       "spend_usd": round(state["spend_usd"], 6)})
        if "per_task" in ho:
            events[-1].update(per_task=ho["per_task"], kept=ho.get("kept", []))
        if ho.get("provider_calls"):
            events[-1]["provider_calls"] = ho["provider_calls"]
    cfg = folder / "config.json"
    providers = (json.loads(cfg.read_text()).get("providers") or []) if cfg.exists() else []
    if providers:
        for e in events:
            e["provider"] = providers[0]
    if strategy:
        for e in events:
            e["strategy"] = strategy
    return events


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder holding the run folders")
    ap.add_argument("--names", nargs="+", help="run folders to take (default: every folder with a state.json)")
    ap.add_argument("--source", default="Part 2 Phase 4 evolve runs (runs/p4/loops, 19 Nov 2026): seed E, splits E, 22-min time box, "
                                        "presets baseline, every-weakness and code-audit, GLM-5.3-Flash on TB2.1")
    ap.add_argument("--out", default=str(Path(__file__).with_name("evolve-recorded.json")))
    a = ap.parse_args(argv)
    names = a.names or sorted(d.name for d in Path(a.runs).iterdir() if (d / "state.json").is_file())
    runs = [{"name": n, "user": n, "events": run_events(Path(a.runs) / n)} for n in names]
    doc = {"source": a.source, "runs": runs}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}: " + ", ".join(f"{r['name']} {len(r['events'])} events" for r in runs))


if __name__ == "__main__":
    main()
