"""Held-out re-check: several agents on a splits file's held-out tasks in ONE HF Job (paired, same batch).

    uv run python devtools/heldout_check.py --splits runs/p2/splits/B.json --trials 4 --out runs/p2/heldout-B.jsonl \
        seed=devtools/agents/seed_b4.py B1=runs/p2/B1/candidates/c001/agent.py B2=runs/p2/B2/candidates/c001/agent.py

Prints each agent's held-out pass rate and its paired difference to the first agent (the seed) with the 95%
task-bootstrap CI (stats.paired_diff). Infra trials do not count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aidev_evolve import cli, launch, stats


def summarize(records: list[dict], names: list[str], tasks: list[str]) -> list[str]:
    base = names[0]
    lines = []
    for n in names:
        pt = stats.per_task(records, n, tasks)
        scored = sum(len(v) for v in pt.values())
        line = f"{n:10} {100 * stats.score(pt):5.1f}% ({sum(sum(v) for v in pt.values())}/{scored} trials)"
        if n != base:
            d = stats.paired_diff(records, n, base, tasks)
            line += f"  vs {base}: {100 * d.diff:+.1f} pts, 95% CI [{100 * d.ci_low:+.1f}, {100 * d.ci_high:+.1f}] over {d.n_tasks} tasks"
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--budget-s", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--summarize-only", action="store_true", help="read --out and print the summary, no Job")
    p.add_argument("agents", nargs="+", help="name=path/to/agent.py; the first is the reference (seed)")
    a = p.parse_args(argv)
    splits = json.loads(Path(a.splits).read_text())
    tasks = splits["heldout"]
    agents = dict(x.split("=", 1) for x in a.agents)
    names = list(agents)
    if not a.summarize_only:
        api, token, user = cli._session()
        spec = {"candidates": {n: Path(path).read_text() for n, path in agents.items()},
                "runs": [{"candidate": n, "tasks": tasks, "trials": a.trials} for n in names], "budget_s": a.budget_s or splits["budget_s"]}
        res = launch.run_eval(api, token, user, spec, on_record=lambda r: print(f"  {r['outcome']:5} {r['candidate']:8} {r['task']}", flush=True))
        with open(a.out, "a") as f:
            for r in res.records:
                f.write(json.dumps(r) + "\n")
        cost = sum(r["cost_usd"] for r in res.records) + res.job_usd
        print(f"Job {res.job_id}: {res.seconds:.0f} s, ${cost:.3f}, {res.missing} missing")
    records = [json.loads(l) for l in Path(a.out).read_text().splitlines() if l.strip()]
    print("\n".join(summarize(records, names, tasks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
