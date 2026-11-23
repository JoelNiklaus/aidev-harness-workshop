"""Scores, the noise margin and the keep-or-drop rule.

The rule (harness-optimization-search): compare child and parent on the same validation tasks; the
parent is re-measured in the same Job as the child (paired, interleaved) and its per-task pass rate pools
every measurement so far, so one lucky run does not become the bar. Keep the child only if its mean pass
rate beats the parent's by the noise margin: max(one task, z x the paired standard error sqrt(d / n)),
with d the flip rate of the seed measured twice (calibration). Infra trials never count.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def per_task(records: list[dict[str, Any]], candidate: str, tasks: list[str] | None = None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if r["candidate"] == candidate and r["outcome"] != "infra" and (tasks is None or r["task"] in tasks):
            out[r["task"]].append(1 if r["outcome"] == "pass" else 0)
    return dict(out)


def score(pt: dict[str, list[int]]) -> float:
    rates = [sum(v) / len(v) for v in pt.values() if v]
    return sum(rates) / len(rates) if rates else 0.0


def flip_rate(pt: dict[str, list[int]]) -> float:
    """The seed's run-to-run noise: the chance that two trials of the same task disagree, averaged over the tasks with
    two or more trials (with exactly two trials: the share of tasks whose trials differ)."""
    rates = [2 * sum(v) * (len(v) - sum(v)) / (len(v) * (len(v) - 1)) for v in pt.values() if len(v) >= 2]
    return sum(rates) / len(rates) if rates else 0.0


def margin(d: float, n: int, z: float = 1.0, k: int = 1) -> float:
    """At least one task; else z paired standard errors of the mean per-task difference when each side's per-task
    rate averages k trials (single-trial flip rate d -> variance about d / k per task)."""
    return max(1 / n, z * math.sqrt(d / (n * k))) if n else 1.0


@dataclass
class Decision:
    keep: bool
    child_score: float
    parent_score: float
    delta: float
    margin: float
    n_tasks: int
    fixed: list[str] = field(default_factory=list)
    broke: list[str] = field(default_factory=list)


def decide(records: list[dict[str, Any]], child: str, parent: str, margin: float, tasks: list[str] | None = None) -> Decision:
    c, p = per_task(records, child, tasks), per_task(records, parent, tasks)
    shared = sorted(set(c) & set(p))
    cr = {t: sum(c[t]) / len(c[t]) for t in shared}
    pr = {t: sum(p[t]) / len(p[t]) for t in shared}
    cs = sum(cr.values()) / len(shared) if shared else 0.0
    ps = sum(pr.values()) / len(shared) if shared else 0.0
    delta = cs - ps
    return Decision(keep=bool(shared) and delta >= margin - 1e-9, child_score=cs, parent_score=ps, delta=delta, margin=margin, n_tasks=len(shared),
                    fixed=[t for t in shared if cr[t] > pr[t]], broke=[t for t in shared if cr[t] < pr[t]])


@dataclass
class PairedDiff:
    diff: float
    ci_low: float
    ci_high: float
    n_tasks: int
    per_task: dict[str, list[float]]  # task -> [rate of b, rate of a]


def paired_diff(records: list[dict[str, Any]], a: str, b: str, tasks: list[str] | None = None, *, n_boot: int = 10_000,
                level: float = 0.95, seed: int = 0) -> PairedDiff:
    """Mean per-task pass-rate difference a - b on the tasks both were scored on, with a percentile bootstrap
    interval that resamples tasks (the unit the claim generalises over). Deterministic (fixed seed)."""
    import random

    pa, pb = per_task(records, a, tasks), per_task(records, b, tasks)
    shared = sorted(set(pa) & set(pb))
    rate = lambda v: sum(v) / len(v)  # noqa: E731
    d = [rate(pa[t]) - rate(pb[t]) for t in shared]
    if not d:
        return PairedDiff(0.0, 0.0, 0.0, 0, {})
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    lo, hi = means[int((1 - level) / 2 * n_boot)], means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return PairedDiff(sum(d) / len(d), lo, hi, len(d), {t: [rate(pb[t]), rate(pa[t])] for t in shared})
