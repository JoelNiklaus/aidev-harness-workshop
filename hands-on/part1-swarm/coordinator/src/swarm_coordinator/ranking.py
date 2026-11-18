"""Rank optimization strategies by held-out gain, so the room ends with a winner that is not noise.

Input: one entry per run with its held-out result (the `heldout` board event, which carries per task
[seed passes, seed trials, final passes, final trials]) and its strategy block (name, preset, hash).

- A run's score is the mean per-task pass rate of its final harness on the fixed held-out tasks.
- A strategy's score is the mean over its runs; its gain is that minus the seed, pooled per task over every run
  in the room (every run measures the same seed on the same tasks, so pooling removes the seed's noise).
- Two strategies tie when the 95% CI of their score difference covers 0. The unit of replication is the run (the
  held-out tasks are the room's fixed set), so the CI is a Welch t interval over run scores; a percentile bootstrap
  over 5 runs under-covers (11% false winners in the simulation of devtools/strategy_noise.py, Welch 4%). `ci_tasks`
  additionally resamples tasks (bootstrap): whether the gain would also hold on other tasks like these.
- Tie groups use the Harness Leaderboard's rule (hlb/scoring.py tie_groups): the best strategy and every
  strategy tied with it form group 1, the best remaining strategy heads group 2, and so on; ranks are shared.
- Strategies with fewer than `min_runs` runs are listed but not ranked (too few runs for a run-level interval with
  useful power: 5 runs per strategy separate an 8-point gap about 70% of the time, 10 runs about 97%).

Pure standard library and deterministic, so the coordinator can vendor this file unchanged.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from typing import Any

MIN_RUNS = 5
N_BOOT = 2000
SEED = 0


def tie_groups(order: list[str], tied: Callable[[str, str], bool]) -> list[list[str]]:
    """Harness Leaderboard rule: head = best not yet grouped; members = the remaining ones tied with the head."""
    out: list[list[str]] = []
    rest = list(order)
    while rest:
        head, *others = rest
        members = [h for h in others if tied(h, head)]
        out.append([head, *members])
        rest = [h for h in others if h not in members]
    return out


def tie_ranks(groups: list[list[str]]) -> list[int]:
    ranks, ahead = [], 0
    for g in groups:
        ranks.append(ahead + 1)
        ahead += len(g)
    return ranks


def t975(df: float) -> float:
    """97.5% quantile of Student's t (Cornish-Fisher expansion; within 0.03 of exact for df >= 2)."""
    z = 1.959963984540054
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2) + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
            + (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / (92160 * df**4))


def welch(a: list[float], b: list[float] | None = None) -> list[float]:
    """95% t interval of mean(a) (b None) or of mean(a) - mean(b) (Welch). Needs 2+ values per side."""
    def mv(x: list[float]) -> tuple[float, float]:
        return _mean(x), (statistics.variance(x) / len(x) if len(x) > 1 else 0.0)

    (ma, va), (mb, vb) = mv(a), (mv(b) if b is not None else (0.0, 0.0))
    se = (va + vb) ** 0.5
    if se == 0:
        return [ma - mb, ma - mb]
    if b is None:
        df = len(a) - 1
    else:
        df = (va + vb) ** 2 / sum(v**2 / (len(x) - 1) for v, x in ((va, a), (vb, b)) if v)
    h = t975(max(df, 1.0)) * se
    return [ma - mb - h, ma - mb + h]


def _ci(values: list[float]) -> list[float]:
    q = statistics.quantiles([round(v, 12) for v in values], n=40, method="inclusive")
    return [q[0], q[-1]]


def _final_rates(run: dict[str, Any]) -> dict[str, float]:
    return {t: v[2] / v[3] for t, v in run["heldout"]["per_task"].items() if v[3]}


def _seed_rates(run: dict[str, Any]) -> dict[str, float]:
    return {t: v[0] / v[1] for t, v in run["heldout"]["per_task"].items() if v[1]}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _run_gain(run: dict[str, Any], n_boot: int, rng: random.Random) -> tuple[float, list[float]]:
    """A single run's own paired gain (final - its seed) with a task-bootstrap CI: shown, never ranked."""
    f, s = _final_rates(run), _seed_rates(run)
    d = [f[t] - s[t] for t in sorted(f.keys() & s.keys())]
    if not d:
        return 0.0, [0.0, 0.0]
    return _mean(d), _ci([_mean([rng.choice(d) for _ in d]) for _ in range(n_boot)])


def rank(runs: list[dict[str, Any]], *, by: str = "hash", min_runs: int = MIN_RUNS, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    runs = [r for r in runs if r.get("heldout") and r["heldout"].get("per_task")]
    if not runs:
        return {"strategies": [], "groups": [], "winner": None, "seed": None, "n_runs": 0, "by": by}
    # the room's seed, pooled per task over every run
    sp, sn = defaultdict(int), defaultdict(int)
    for r in runs:
        for t, v in r["heldout"]["per_task"].items():
            sp[t] += v[0]
            sn[t] += v[1]
    tasks = sorted(t for t in sn if sn[t])
    seed_rate = {t: sp[t] / sn[t] for t in tasks}
    room_seed = _mean(list(seed_rate.values()))

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        members[r["strategy"][by] if by in r["strategy"] else r["strategy"]["hash"]].append(r)
    rng = random.Random(seed)
    finals = {k: [_final_rates(r) for r in rs] for k, rs in members.items()}
    scores = {k: [_mean([fr[t] - seed_rate[t] for t in tasks if t in fr]) + room_seed for fr in frs] for k, frs in finals.items()}

    def boot_tasks(k: str) -> list[float]:
        frs = finals[k]
        r = random.Random(f"{seed}:tasks:{k}")
        out = []
        for _ in range(n_boot):
            ts = [r.choice(tasks) for _ in tasks]
            rs = [r.choice(frs) for _ in frs]
            out.append(_mean([_mean([fr[t] - seed_rate[t] for t in ts if t in fr]) for fr in rs]))  # gain over the pooled seed
        return out

    entries = []
    for k, rs in members.items():
        st = rs[0]["strategy"]
        score = _mean(scores[k])
        ranked = len(rs) >= min_runs
        entries.append({"key": k, "name": st.get("name", k), "preset": st.get("preset"), "hash": st.get("hash"), "n_runs": len(rs),
                        "score": score, "gain": score - room_seed, "ci": [c - room_seed for c in welch(scores[k])] if len(rs) > 1 else None, "ci_tasks": _ci(boot_tasks(k)),
                        "ranked": ranked, "rank": None, "group": None, "runs": [r["run_id"] for r in rs],
                        "note": "" if ranked else f"needs {min_runs} runs to be ranked (has {len(rs)})"})
    entries.sort(key=lambda e: (not e["ranked"], -e["score"], e["key"]))
    order = [e["key"] for e in entries if e["ranked"]]

    def tied(a: str, b: str) -> bool:
        lo, hi = welch(scores[a], scores[b])
        return lo <= 0 <= hi

    groups = tie_groups(order, tied)
    by_key = {e["key"]: e for e in entries}
    for gi, (g, rk) in enumerate(zip(groups, tie_ranks(groups))):
        for k in g:
            by_key[k]["rank"], by_key[k]["group"] = rk, gi
    winner = None
    if groups:
        top = groups[0]
        best = max((r for k in top for r in members[k]), key=lambda r: (_mean(list(_final_rates(r).values())), r["run_id"]))
        gain, ci = _run_gain(best, n_boot, rng)
        winner = {"keys": top, "names": [by_key[k]["name"] for k in top], "tie": len(top) > 1,
                  "patch": {"run_id": best["run_id"], "user": best.get("user"), "strategy": best["strategy"].get("name"),
                            "kept": list(best.get("kept") or []), "best_candidate": best["heldout"].get("best_candidate"),
                            "gain": gain, "ci": ci}}
    return {"strategies": entries, "groups": groups, "winner": winner, "seed": room_seed, "n_runs": len(runs), "by": by}


def format_table(result: dict[str, Any]) -> str:
    """Plain-text ranking for the terminal (`aidev-evolve rank`)."""
    if not result["strategies"]:
        return "No run has a held-out result yet."
    pts = lambda x: f"{100 * x:+.1f}"  # noqa: E731
    lines = [f"Held-out seed (pooled over {result['n_runs']} runs): {100 * result['seed']:.1f}%", "",
             f"{'rank':>4}  {'strategy':24} {'runs':>4}  {'final':>6}  {'gain':>6}  95% CI (runs)   95% CI (runs+tasks)"]
    for e in result["strategies"]:
        rk = f"={e['rank']}" if e["rank"] and sum(1 for x in result["strategies"] if x["rank"] == e["rank"]) > 1 else (str(e["rank"]) if e["rank"] else "-")
        ci = f"[{pts(e['ci'][0])}, {pts(e['ci'][1])}]" if e["ci"] else "[n/a]"
        lines.append(f"{rk:>4}  {e['name'][:24]:24} {e['n_runs']:>4}  {100 * e['score']:5.1f}%  {pts(e['gain']):>6}  "
                     f"{ci:15} [{pts(e['ci_tasks'][0])}, {pts(e['ci_tasks'][1])}]  {e['note']}")
    w = result["winner"]
    if w:
        lines.append("")
        lines.append(("Tie for first: " + ", ".join(w["names"]) + " (their difference is within noise)") if w["tie"] else f"Winning strategy: {w['names'][0]}")
        p = w["patch"]
        lines.append(f"Best run in that group: {p['run_id']} ({p['strategy']}), held-out {pts(p['gain'])} pts vs its seed, "
                     f"95% CI [{pts(p['ci'][0])}, {pts(p['ci'][1])}]; kept: {'; '.join(p['kept']) or 'nothing'}")
    return "\n".join(lines)
