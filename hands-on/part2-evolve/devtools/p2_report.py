"""Phase 2 report: config comparison, held-out results and the two slide figures, from the run folders.

    uv run --with matplotlib python devtools/p2_report.py --runs A1,A2,B1,B2 --out "<talk>/evolve"

Writes `<out>/p2-numbers.md` (every number the gallery and the report quote), `<out>/p2-validation.png`
(validation pass rate of the current kept harness over iterations, pooled over its measurements, so it can drop on a re-measure; one line per run) and `<out>/p2-heldout.png` (held-out: seed vs
the run's final agent, paired difference with its 95% task-bootstrap CI). Figures are drawn at the size they
take on a 13.33 x 7.5 in slide, every text >= 24 pt, through figure_check.save_checked.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "p2"
CONFIG_LABEL = {"A": "A: Phase 1 seed, 240 s", "B": "B: 4-weakness seed, 180 s", "C": "C"}
COLOR = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
MARKER = {"1": "o", "2": "s", "3": "^"}
SLIDE_SIZE = (11.4, 5.6)
FONT = 24


def load(run: str) -> dict:
    d = RUNS / run
    return {"name": run, "state": json.loads((d / "state.json").read_text()), "config": json.loads((d / "config.json").read_text())}


def run_row(r: dict) -> dict:
    st, h = r["state"], r["state"]["history"]
    evaluated = [x for x in h if "val_child" in x]
    ho = st.get("heldout") or {}
    return {"run": r["name"], "budget": r["config"]["budget_s"], "iterations": len(h), "kept": sum(x["verdict"] == "kept" for x in h),
            "dropped_on_remeasure": sum(x["verdict"].startswith("dropped: did not hold") for x in h),
            "seed_val": st["curve"][0], "final_val": st["curve"][-1], "final": st["incumbent"],
            "iter_min_median": statistics.median(x["iter_s"] for x in evaluated) / 60 if evaluated else None,
            "iter_usd_mean": statistics.mean(x["iter_usd"] for x in evaluated) if evaluated else None,
            "proposer_s_median": statistics.median(x["proposer_s"] for x in h) if h else None,
            "spend": st["spend_usd"], "proposer_usd": st["proposer_usd"], "heldout": ho}


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def heldout_records(config: str, runs: list[dict]) -> list[dict]:
    """Every held-out trial of a config, candidates renamed to 'seed' or the run name: the in-loop check of each
    run (seed c000 vs its final, 2 trials) plus the re-check Job (all finals and the seed in one batch)."""
    out = []
    for r in runs:
        if r["name"][0] != config:
            continue
        final = r["state"]["incumbent"]
        for line in (RUNS / r["name"] / "records.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x.get("eval") == "heldout" and x["candidate"] in ("c000", final):
                out.append({**x, "candidate": "seed" if x["candidate"] == "c000" else r["name"], "source": f"in-loop {r['name']}"})
    recheck = RUNS / f"heldout-{config}.jsonl"
    if recheck.exists():
        out += [{**json.loads(l), "source": "re-check"} for l in recheck.read_text().splitlines() if l.strip()]
    return out


def trial_bootstrap(records: list[dict], a: str, b: str, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float, int, int]:
    """Mean per-task difference a - b with a CI that resamples trials within each task (the tasks are fixed: is
    the gain on THESE held-out tasks real?). Returns diff, low, high, n_tasks, trials of a."""
    import random

    from aidev_evolve import stats

    pa, pb = stats.per_task(records, a), stats.per_task(records, b)
    tasks = sorted(set(pa) & set(pb))
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    diff = mean([mean(pa[t]) - mean(pb[t]) for t in tasks])
    rng = random.Random(seed)
    boots = sorted(mean([mean([rng.choice(pa[t]) for _ in pa[t]]) - mean([rng.choice(pb[t]) for _ in pb[t]]) for t in tasks]) for _ in range(n_boot))
    return diff, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], len(tasks), sum(len(pa[t]) for t in tasks)


def heldout_table(runs: list[dict]) -> tuple[str, list[dict]]:
    from aidev_evolve import stats

    lines = ["| Config | Agent | Held-out pass (trials) | Gain over seed | 95% CI, tasks resampled | 95% CI, trials resampled |", "|---|---|---|---|---|---|"]
    rows = []
    for cfg in sorted({r["name"][0] for r in runs}):
        recs = heldout_records(cfg, runs)
        names = ["seed"] + [r["name"] for r in runs if r["name"][0] == cfg] + (["ref"] if any(x["candidate"] == "ref" for x in recs) else [])
        for n in names:
            pt = stats.per_task(recs, n)
            k = sum(len(v) for v in pt.values())
            row = {"config": cfg, "agent": n, "pass": stats.score(pt), "trials": k}
            if n == "seed":
                lines.append(f"| {cfg} | seed | {pct(row['pass'])} ({k}) | | | |")
            else:
                d = stats.paired_diff(recs, n, "seed")
                td = trial_bootstrap(recs, n, "seed")
                row.update(diff=d.diff, task_ci=(d.ci_low, d.ci_high), trial_ci=(td[1], td[2]))
                label = "hand-fixed reference" if n == "ref" else f"{n} final"
                lines.append(f"| {cfg} | {label} | {pct(row['pass'])} ({k}) | {100 * d.diff:+.0f} pts | [{100 * d.ci_low:+.0f}, {100 * d.ci_high:+.0f}] | "
                             f"[{100 * td[1]:+.0f}, {100 * td[2]:+.0f}] |")
            rows.append(row)
    return "\n".join(lines) + "\n", rows


def unseen_table(runs: list[dict]) -> str:
    """The wider check: the seed and each run's final on every pool task the loop never saw (one Job per config)."""
    from aidev_evolve import stats

    out = []
    for cfg in sorted({r["name"][0] for r in runs}):
        path = RUNS / f"heldout-{cfg}-unseen.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        n_tasks = len(stats.per_task(recs, "seed"))
        out += [f"### Config {cfg}: all {n_tasks} pool tasks the loop never saw (re-check Job, every agent in one batch)", "",
                "| Agent | Pass (trials) | Gain over seed | 95% CI, tasks resampled |", "|---|---|---|---|"]
        for n in ["seed"] + [r["name"] for r in runs if r["name"][0] == cfg]:
            pt = stats.per_task(recs, n)
            k = sum(len(v) for v in pt.values())
            if n == "seed":
                out.append(f"| seed | {pct(stats.score(pt))} ({k}) | | |")
            else:
                d = stats.paired_diff(recs, n, "seed")
                out.append(f"| {n} final | {pct(stats.score(pt))} ({k}) | {100 * d.diff:+.0f} pts | [{100 * d.ci_low:+.0f}, {100 * d.ci_high:+.0f}] |")
        out.append("")
    return "\n".join(out)


def fmt(x: float | None, f: str) -> str:
    return "-" if x is None else f.format(x)


def numbers_md(rows: list[dict]) -> str:
    out = ["# Phase 2 numbers (generated by devtools/p2_report.py; do not edit)", "",
           "| Run | Budget | Iterations | Kept | Dropped on re-measure | Validation seed -> final | Median iteration | Mean $/iteration | Run spend (HF) | Held-out seed -> final | Paired diff [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        ho = x["heldout"]
        hos = f"{pct(ho.get('seed'))} -> {pct(ho.get('best'))} ({ho.get('best_candidate')}, {ho.get('n_tasks')} tasks x {ho.get('trials')})" if ho else "-"
        ci = f"{100 * ho['diff']:+.0f} [{100 * ho['ci_low']:+.0f}, {100 * ho['ci_high']:+.0f}]" if ho else "-"
        out.append(f"| {x['run']} | {x['budget']:.0f} s | {x['iterations']} | {x['kept']} | {x['dropped_on_remeasure']} | {pct(x['seed_val'])} -> {pct(x['final_val'])} ({x['final']}) | "
                   f"{fmt(x['iter_min_median'], '{:.1f} min')} | {fmt(x['iter_usd_mean'], '${:.2f}')} | ${x['spend']:.2f} | {hos} | {ci} |")
    return "\n".join(out) + "\n"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": FONT, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_validation(runs: list[dict], path: Path) -> None:
    plt = _style()
    from figure_check import save_checked

    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    for r in runs:
        cfg, k = r["name"][0], r["name"][1:]
        curve = [100 * c for c in r["state"]["curve"]]
        ax.step(range(len(curve)), curve, where="post", color=COLOR[cfg], lw=3, marker=MARKER.get(k, "o"), ms=11,
                label=r["name"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Current harness:\nvalidation pass (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, max(len(r["state"]["curve"]) for r in runs), 2))
    ax.grid(axis="y", color="#E2E8F0")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.36)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=4, frameon=False, fontsize=FONT, handlelength=1.5, columnspacing=1.2)
    save_checked(fig, path, display_width_in=SLIDE_SIZE[0])


def unseen_rows(runs: list[dict]) -> list[dict]:
    from aidev_evolve import stats

    rows = []
    for cfg in sorted({r["name"][0] for r in runs}):
        path = RUNS / f"heldout-{cfg}-unseen.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        seed = stats.score(stats.per_task(recs, "seed"))
        for r in runs:
            if r["name"][0] == cfg:
                d = stats.paired_diff(recs, r["name"], "seed")
                rows.append({"config": cfg, "agent": r["name"], "seed": seed, "pass": stats.score(stats.per_task(recs, r["name"])),
                             "diff": d.diff, "ci": (d.ci_low, d.ci_high), "n_tasks": d.n_tasks})
    return rows


def fig_heldout(rows: list[dict], path: Path) -> None:
    """One message: the loop's final agents of config B beat their seed on every task the loop never saw; config A's do
    not. Gain over the seed per run's final agent; whiskers: 95% CI, tasks resampled."""
    plt = _style()
    from figure_check import save_checked

    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["ci"]
        ax.errorbar([100 * r["diff"]], [y], xerr=[[100 * (r["diff"] - lo)], [100 * (hi - r["diff"])]], fmt=MARKER.get(r["agent"][1:], "o"),
                    color=COLOR[r["config"]], ms=14, lw=3, capsize=8)
    ax.axvline(0, color="#595959", lw=2, ls="--", label="seed agent (no gain)")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['agent']}  {pct(r['seed'])}→{pct(r['pass'])}" for r in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Gain on unseen tasks (points)")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color="#E2E8F0")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.36)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.33), frameon=False, fontsize=FONT)
    save_checked(fig, path, display_width_in=SLIDE_SIZE[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="A1,A2,B1,B2")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    runs = [load(r) for r in a.runs.split(",") if (RUNS / r / "state.json").exists()]
    out = Path(a.out)
    out.mkdir(parents=False, exist_ok=True)
    rows = [run_row(r) for r in runs]
    ho_md, ho_rows = heldout_table(runs)
    (out / "p2-numbers.md").write_text(numbers_md(rows) + "\n## Held-out, every measurement pooled (in-loop check + re-check Job)\n\n" + ho_md
                                       + "\n## Wider check\n\n" + unseen_table(runs))
    fig_validation(runs, out / "p2-validation.png")
    if unseen_rows(runs):
        fig_heldout(unseen_rows(runs), out / "p2-heldout.png")
    print((out / "p2-numbers.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
