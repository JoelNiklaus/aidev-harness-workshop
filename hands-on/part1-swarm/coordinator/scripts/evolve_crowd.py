"""Fill a local /evolve board with SYNTHETIC attendees, to check the projector page under a full room.

    uv run coordinator --fake-auth --port 8772 &
    uv run python scripts/evolve_crowd.py --url http://127.0.0.1:8772 --attendees 300 --iterations 6
    uv run scripts/board_check.py --url http://127.0.0.1:8772/evolve --svg race --shot /tmp/evolve-crowd.png

Only for a --fake-auth coordinator (the bearer token is the username). The numbers are made up: seeds around
40%, most runs keep a time-budget fix in iteration 1 and drop later patches, like the recorded runs.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_coordinator.evolve import SCHEMA  # noqa: E402
from swarm_coordinator.loadtest import EVOLVE_HEADERS  # noqa: E402

TITLES = [("Run until the time budget, not 10 steps", "the loop stops at the 10-step cap with most of the 180 s left."),
          ("Show stderr to the model", "stderr goes to /dev/null, so failing commands show no error message."),
          ("Keep head and tail of long outputs", "the model sees only the first 1,000 chars; errors at the end are lost."),
          ("Verify the result before finishing", "runs end with a text answer that was never executed."),
          ("Show the clock after every command", "the model never sees how much time is left."),
          ("Add a per-command timeout", "a hung command eats the whole budget."),
          ("Detect repeated commands, force a new approach", "runs retry the same failing command 20 times.")]


def events(user: str, iterations: int, rng: random.Random) -> list[dict]:
    rid = f"{user}-run"
    best = round(rng.uniform(0.3, 0.5), 3)
    spend = 0.11
    out = [{"schema": SCHEMA, "run_id": rid, "kind": "seed", "val_score": best, "search_score": 0.0, "spend_usd": spend}]
    for i in range(1, iterations + 1):
        title, hyp = TITLES[0] if i == 1 and rng.random() < 0.7 else rng.choice(TITLES[1:])
        child = min(1.0, best + (rng.uniform(0.3, 0.5) if title == TITLES[0][0] else rng.uniform(-0.1, 0.12)))
        kept = child - best >= 0.1
        parent = best
        best = child if kept else max(0.0, best + rng.uniform(-0.02, 0.01))
        spend += rng.uniform(0.12, 0.2)
        diff = "--- agent.py\n+++ agent.py\n" + "".join(f"+changed line {k}\n" for k in range(rng.randint(1, 12))) + "-old line\n"
        out.append({"schema": SCHEMA, "run_id": rid, "kind": "iteration", "iteration": i, "candidate": f"c{i:03d}", "parent": "c000",
                    "title": title, "hypothesis": "Hypothesis: " + hyp, "accepted": kept, "verdict": "kept" if kept else "dropped",
                    "val_child": round(child, 3), "val_parent": parent, "best_val": round(best, 3), "fixed": [], "broke": [],
                    "diff": diff, "spend_usd": round(spend, 4)})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--attendees", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    with httpx.Client(base_url=a.url, timeout=30) as c:
        names = ["ada", "grace", "linus", "margaret", "barbara", "ken", "dennis", "alan", "edsger", "donald"]
        for k in range(a.attendees):
            user = f"{names[k % len(names)]}{k}"
            for e in events(user, rng.randint(1, a.iterations), rng):
                c.post("/api/evolve/submit", json=e, headers={"Authorization": f"Bearer {user}", **EVOLVE_HEADERS}).raise_for_status()
    print(f"posted {a.attendees} synthetic attendees to {a.url}")


if __name__ == "__main__":
    main()
