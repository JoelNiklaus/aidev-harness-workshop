"""Phase 4 splits (config E): which seed weakness bites which pool task, and the chosen splits.

    uv run python devtools/splits_e.py --out src/aidev_evolve/data/splits.json   # needs runs/p4 and runs/p3 sweeps

Paired sweeps on the 34-task pool, 4 trials per agent (2 Jobs x 2 trials each, all agents interleaved per Job):
`fixed` = seed E with all three weaknesses fixed (devtools/agents/c_steps_loop.py: the step fix as the loops find it;
seed C has neither of the two new weaknesses), `w2only` / `w3only` = only the stray-process kill / only the task cut
left (e_w2_only.py, e_w3_only.py), `seedC` = only the step limit (and the 10-s kill) left, `seedE` = all three.
A weakness bites a task when fixing it adds >= 2 of 4 passes.

The split rule (decided before the loops ran):
- search: tasks whose seed transcripts show each weakness, preferring model-limited long tasks (they cost no headroom
  elsewhere). The stray-process kill shows in every command of every transcript (the cleanup is part of the command)
  and the digest flags a cut task on step 1, so they need no extra search tasks.
- validation: >= 2 tasks per weakness and 1 guard, so the fix of each weakness can clear the keep margin.
- held-out: every other pool task, with tasks for each weakness left in it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = {"runs/p4/sweep-E.jsonl": {"steps10": "fixed"}, "runs/p4/sweep-D.jsonl": {"steps10": "fixed"}, "runs/p3/seedC.jsonl": {}}
SEARCH = ["count-dataset-tokens", "break-filter-js-from-html", "bn-fit-modify", "sparql-university", "configure-git-webserver"]
VALIDATION = ["vulnerable-secret", "sqlite-with-gcov", "fix-code-vulnerability", "pypi-server", "kv-store-grpc",
              "multi-source-data-merger", "log-summary-date-ranges", "fix-git"]
LEFT = {"step limit": "seedC", "stray-process kill": "w2only", "task cut": "w3only"}  # weakness -> the agent that has only it left
COMMENT = ("Fixed splits for aidev-evolve (Phase 4, config E): Terminal-Bench 2.1, GLM-5.3-Flash effort low, 180 s agent budget, "
           "seed E = seed C (10-step limit, 10-s command kill, stderr hidden, 1,000-char output head) + every process a command leaves "
           "running is killed after it + the task text cut at 1,000 chars. Chosen 2026-11-19 by devtools/splits_e.py from paired sweeps "
           "on the 34-task pool, 4 trials each (runs/p4/sweep-E-*.jsonl, sweep-D-*.jsonl, runs/p3/seedC.jsonl); the rule is in its "
           "docstring. Calibration per task: seed (seed E), fixed (all three weaknesses fixed), w2_left / w3_left (only that weakness "
           "left), seed_c (only the step limit left), bites (weaknesses whose fix adds >= 2 of 4 passes). Held-out = every other pool "
           "task (21).")


def counts(root: Path = ROOT) -> dict[str, dict[str, list[int]]]:
    d: dict[str, dict[str, list[int]]] = {}
    for f, rename in SWEEPS.items():
        for line in (root / f).read_text().splitlines():
            r = json.loads(line)
            if r["outcome"] not in ("pass", "fail"):
                continue
            a = d.setdefault(r["task"], {}).setdefault(rename.get(r["candidate"], r["candidate"]), [0, 0])
            a[0] += r["outcome"] == "pass"
            a[1] += 1
    return d


def build(d: dict[str, dict[str, list[int]]]) -> dict:
    rate = lambda t, c: d[t][c][0] / d[t][c][1]  # noqa: E731
    pool = sorted(d)
    heldout = [t for t in pool if t not in SEARCH + VALIDATION]
    cal = {t: {"seed": "%d/%d" % tuple(d[t]["seedE"]), "fixed": "%d/%d" % tuple(d[t]["fixed"]), "w2_left": "%d/%d" % tuple(d[t]["w2only"]),
               "w3_left": "%d/%d" % tuple(d[t]["w3only"]), "seed_c": "%d/%d" % tuple(d[t]["seedC"]),
               "bites": [w for w, c in LEFT.items() if rate(t, "fixed") - rate(t, c) >= 0.5]} for t in pool}
    return {"_comment": COMMENT, "benchmark": "terminal-bench-2.1", "task_commit": "d49e28f1e4ddd13d289e85a5f312a66750951932",
            "budget_s": 180.0, "search": SEARCH, "validation": VALIDATION, "heldout": heldout, "calibration": cal}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(json.dumps(build(counts()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
