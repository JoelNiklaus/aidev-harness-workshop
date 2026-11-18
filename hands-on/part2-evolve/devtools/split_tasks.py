"""Phase 2 task selection: which tasks can a harness patch fix inside the time budget?

    uv run python devtools/split_tasks.py SEED.jsonl REF.jsonl [--out splits.json --budget 180 --name B]

The rule (written once, before any loop runs):
- Pool: the 34 fast Terminal-Bench 2.1 tasks of the Phase 1 sweep (headless-terminal removed: its verifier hangs).
- Measure the seed and a hand-fixed reference harness (devtools/agents/reference.py: the seed with every known
  weakness fixed) 4 times each per task, at the configuration's time budget.
- headroom: the reference passes >= 3 of 4 and at least 2 of 4 more than the seed (pass-rate gap >= 0.5). The
  harness, not the model, decides these tasks, and a patch can convert them within the budget. (The stricter
  "seed <= 1/4" found only 0-1 tasks in the pool: GLM-5.3-Flash passes most fast tasks with any loop.)
- guard: seed and reference both pass every trial (a patch that breaks working behaviour shows up here).
- model-limited (reference <= 1 of 4) and noisy tasks (everything else) are left out: they add noise, not signal.
- Headroom tasks go to validation and held-out in turn (held-out first), sorted by the seed's failure mechanism, so
  both see each mechanism. 2 guards go to validation, up to 4 to held-out.
- Search (the proposer reads its traces; its score decides nothing): up to 4 tasks the seed fails at least half the
  time that are neither headroom nor guards, largest reference-minus-seed gap first. Their failures show the
  weaknesses; whether the model can then solve them does not matter.
Infra trials do not count. Prints the per-task table and writes the splits file.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TASK_COMMIT = "d49e28f1e4ddd13d289e85a5f312a66750951932"


def load(path: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["outcome"] != "infra":
                by[r["task"]].append(r)
    return by


def mechanism(rec: dict, cap: int = 1000) -> str:
    """Why a failed seed trial failed, read from its trajectory."""
    t = rec.get("trajectory") or {}
    steps = t.get("steps", [])
    shell = [s for s in steps if s["kind"] == "shell"]
    if t.get("end") == "timeout":
        longest = max((s.get("duration_s") or 0 for s in shell), default=0)
        return "hung command" if longest > 60 or (shell and shell[-1].get("timed_out")) else "out of time"
    if t.get("end") == "error":
        return "agent crash"
    cut = sum((s.get("output_chars") or 0) > cap for s in shell)
    if cut >= 2:
        return "output cut"
    return "stopped early"


def table(seed: dict, ref: dict) -> list[dict]:
    rows = []
    for task in sorted(set(seed) | set(ref)):
        s, r = seed.get(task, []), ref.get(task, [])
        sp, rp = sum(x["outcome"] == "pass" for x in s), sum(x["outcome"] == "pass" for x in r)
        mechs = [mechanism(x) for x in s if x["outcome"] == "fail"]
        main = max(set(mechs), key=mechs.count) if mechs else "-"
        if len(s) >= 3 and len(r) >= 3 and rp / len(r) >= 0.75 and rp / len(r) - sp / len(s) >= 0.5 - 1e-9:
            kind = "headroom"
        elif s and r and sp == len(s) and rp == len(r):
            kind = "guard"
        elif r and rp <= 1:
            kind = "model-limited"
        else:
            kind = "noisy"
        rows.append({"task": task, "seed": f"{sp}/{len(s)}", "ref": f"{rp}/{len(r)}", "kind": kind, "mechanism": main,
                     "seed_s": round(sum(x["agent_s"] or 0 for x in s) / max(1, len(s))), "ref_s": round(sum(x["agent_s"] or 0 for x in r) / max(1, len(r)))})
    return rows


def deal(rows: list[dict]) -> dict[str, list[str]]:
    head = sorted((x for x in rows if x["kind"] == "headroom"), key=lambda x: (x["mechanism"], x["task"]))
    guards = sorted(x["task"] for x in rows if x["kind"] == "guard")
    rate = lambda s: int(s.split("/")[0]) / max(1, int(s.split("/")[1]))  # noqa: E731
    readable = sorted((x for x in rows if x["kind"] in ("noisy", "model-limited") and rate(x["seed"]) <= 0.5 and x["ref"] != "0/0"),
                      key=lambda x: (rate(x["seed"]) - rate(x["ref"]), x["task"]))
    out: dict[str, list[str]] = {"search": [x["task"] for x in readable[:4]], "validation": [], "heldout": []}
    for i, x in enumerate(head):
        out["heldout" if i % 2 == 0 else "validation"].append(x["task"])
    out["validation"] += guards[:2]
    out["heldout"] += guards[2:6]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("seed")
    p.add_argument("ref")
    p.add_argument("--out", default=None)
    p.add_argument("--budget", type=float, default=180)
    p.add_argument("--name", default="")
    a = p.parse_args(argv)
    rows = table(load(a.seed), load(a.ref))
    print(f"{'task':32} seed  ref   kind           seed mechanism  seed_s ref_s")
    for x in sorted(rows, key=lambda x: (x["kind"], x["task"])):
        print(f"{x['task']:32} {x['seed']:5} {x['ref']:5} {x['kind']:14} {x['mechanism']:15} {x['seed_s']:5} {x['ref_s']:5}")
    splits = deal(rows)
    for k, v in splits.items():
        print(k, len(v), v)
    if a.out:
        calib = {x["task"]: {"seed": x["seed"], "ref": x["ref"], "kind": x["kind"], "mechanism": x["mechanism"]} for x in rows}
        Path(a.out).write_text(json.dumps({
            "_comment": f"Phase 2 splits {a.name} (budget {a.budget:.0f} s), from 4 trials each of the seed and the hand-fixed reference "
                        "harness (devtools/split_tasks.py has the rule): headroom = seed <= 1/4 and reference >= 3/4; guard = both 4/4.",
            "benchmark": "terminal-bench-2.1", "task_commit": TASK_COMMIT, "budget_s": a.budget, **splits, "calibration": calib}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
