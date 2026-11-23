"""Optimization strategies: the part of the evolve loop an attendee may edit.

A strategy is a folder with `strategy.toml` (the knobs) and `proposer_prompt.md` (how the coding agent should
approach a proposal). Bundled presets live in `aidev_evolve/strategies/<name>/`; `aidev-evolve strategy-init
<preset> <dir>` copies one for editing, `aidev-evolve run --strategy <preset|dir>` runs it.

Editable (they change what the loop finds, none can break the evaluation):
- evidence: which search transcripts the proposer reads (`traces` failed|all, `max_traces`), how much of the
  earlier attempts it sees (`history` none|titles|full) and whether their diffs are included (`diffs`);
- proposer: its approach (the prompt file) and `candidates` per iteration (1, or 2 screened on validation, the
  better one goes through the keep rule; `second_hint` steers the second proposal);
- keep rule: validation trials per task (`val_trials`), the margin in paired standard errors (`z`) and the
  re-measure of a winner (`confirm`; always on with 2 candidates, since picking the better of two is a selection);
- `iterations`: the most iterations of `run --full` (its time box decides how many start); the workshop run does one.

Fixed for every attendee, so strategies compete fairly (refused if a strategy file names them): the model and
its settings, the per-task time budget, the task splits and the held-out set, the seed agent, the held-out
trials, the shape of the run (workshop or --full) and the credit cap. The fixed rules of the proposer prompt (contract, allowed imports,
PATCH.md format) are appended by the loop and cannot be edited either.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tomllib
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT = "code-audit"  # the preset `run` uses: the Phase 4 winner (19 Nov, 20 real runs: PROGRESS.md)
SESSION_MINUTES = 22.0  # the room's time box per run: seed calibration + iterations + held-out check
USD_CAP = 2.0  # worst-case HF credits per run (every iteration re-measured)
HELDOUT_TRIALS = 2  # `run --full`: trials per held-out task for the seed and the final agent
# The workshop run (the default of `run`; DESIGN.md, Phase 5): the seed's scores ship in the wheel, one iteration, and one
# Job measures validation and held-out together. One trial per held-out task: the board pools the runs of a block.
WORKSHOP_ITERATIONS = 1
WORKSHOP_HELDOUT_TRIALS = 1
# In the room each seating block runs one preset (`run --block left|middle|right`, or A|B|C), so every preset gets
# enough runs to be ranked (the board ranks a strategy from 5 runs); with one default nearly everyone would run it.
BLOCKS = {"left": "baseline", "middle": "every-weakness", "right": "code-audit"}
BLOCK_LETTERS = dict(zip("abc", BLOCKS))


def block_strategy(block: str) -> str:
    """The preset of a seating block: left|middle|right or A|B|C, any case."""
    key = block.strip().lower()
    key = BLOCK_LETTERS.get(key, key)
    if key not in BLOCKS:
        raise ValueError(f"no block {block!r}: choose from {', '.join(BLOCKS)} (or A, B, C)")
    return BLOCKS[key]

# Measured on config B (PROGRESS.md, Phase 2): minutes of wall time and HF credits per trial and per Job.
CALIB_MIN, ITER_MIN, ITER_MIN_TWO, REMEASURE_MIN, HELDOUT_MIN = 3.0, 4.2, 4.6, 2.5, 5.5
USD_PER_TRIAL, USD_PER_JOB = 0.005, 0.01
# The workshop run: the coding agent's proposal, then one Job (alone: about 5 min; with 40 loops at once: see PROGRESS.md, 23 Nov).
WORKSHOP_PROPOSE_MIN, WORKSHOP_JOB_MIN = 1.5, 5.0

FIXED = ("model", "effort", "provider", "budget_s", "splits", "search", "validation", "heldout", "seed", "heldout_trials",
         "session_minutes", "usd_cap", "benchmark")
TABLES = {"evidence": {"traces", "max_traces", "history", "diffs"}, "proposer": {"candidates", "prompt_file", "second_hint"},
          "keep": {"val_trials", "z", "confirm"}}
TOP = {"name", "based_on", "description", "iterations"}
CHOICES = {"traces": ("failed", "all"), "history": ("none", "titles", "full")}
RANGES = {"max_traces": (1, 8), "candidates": (1, 2), "val_trials": (1, 3), "z": (0.0, 3.0), "iterations": (1, 6)}
PROMPT_MAX = 4000


class StrategyError(ValueError):
    pass


@dataclass(frozen=True)
class Strategy:
    name: str
    preset: str
    description: str
    prompt: str
    traces: str = "all"
    max_traces: int = 8
    history: str = "full"
    diffs: bool = False
    candidates: int = 1
    second_hint: str = ""
    val_trials: int = 2
    z: float = 1.5
    confirm: bool = True
    iterations: int = 4

    def knobs(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("traces", "max_traces", "history", "diffs", "candidates", "val_trials", "z", "confirm", "iterations")}

    @property
    def hash(self) -> str:
        """Identity on the board: same knobs and same prompt = same strategy, whoever runs it."""
        blob = json.dumps({"knobs": self.knobs(), "prompt": self.prompt.strip(), "second_hint": self.second_hint.strip()}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @property
    def modified(self) -> bool:
        return self.preset not in presets() or self.hash != _preset_hash(self.preset)

    @property
    def remeasure(self) -> bool:
        return self.confirm or self.candidates > 1

    def event(self) -> dict[str, Any]:
        return {"name": self.name, "preset": self.preset, "hash": self.hash, "modified": self.modified, "knobs": self.knobs(), "prompt": self.prompt[:PROMPT_MAX]}


def _root() -> Any:
    return resources.files("aidev_evolve.strategies")


def presets() -> list[str]:
    return sorted(p.name for p in _root().iterdir() if p.is_dir() and p.joinpath("strategy.toml").is_file())


_HASHES: dict[str, str] = {}


def _preset_hash(name: str) -> str:
    if name not in _HASHES:
        p = _root().joinpath(name)
        _HASHES[name] = _parse(p.joinpath("strategy.toml").read_text(), lambda f: p.joinpath(f).read_text(), name).hash
    return _HASHES[name]


def _check(key: str, value: Any) -> None:
    if key in CHOICES and value not in CHOICES[key]:
        raise StrategyError(f"{key} = {value!r}: choose one of {', '.join(CHOICES[key])}")
    if key in RANGES:
        lo, hi = RANGES[key]
        number = (float, int) if key == "z" else (int,)
        if isinstance(value, bool) or not isinstance(value, number) or not lo <= value <= hi:
            raise StrategyError(f"{key} = {value!r}: allowed {lo}..{hi}{'' if key == 'z' else ' (whole number)'}")
    if key in ("diffs", "confirm") and not isinstance(value, bool):
        raise StrategyError(f"{key} = {value!r}: true or false")


def _parse(text: str, read: Any, default_preset: str) -> Strategy:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise StrategyError(f"strategy.toml: {e}") from e
    tables = [("", data)] + [(k, v) for k, v in data.items() if isinstance(v, dict)]
    for where, table in tables:
        for key in table:
            if key in FIXED:
                raise StrategyError(f"{key} is fixed for every attendee (fairness): the model, time budget, splits, held-out set, seed and "
                                    f"session length are the same for all strategies; remove `{key}`{f' from [{where}]' if where else ''}")
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in TABLES:
                raise StrategyError(f"unknown section [{key}] (known: {', '.join(TABLES)})")
            for k in value:
                if k not in TABLES[key]:
                    raise StrategyError(f"unknown knob {k!r} in [{key}] (known: {', '.join(sorted(TABLES[key]))})")
        elif key not in TOP:
            raise StrategyError(f"unknown setting {key!r} (known: {', '.join(sorted(TOP))} and the sections {', '.join(TABLES)})")
    ev, pr, kp = data.get("evidence", {}), data.get("proposer", {}), data.get("keep", {})
    knobs = {**ev, **{k: v for k, v in pr.items() if k != "prompt_file"}, **kp}
    if "iterations" in data:
        knobs["iterations"] = data["iterations"]
    for k, v in knobs.items():
        _check(k, v)
    if "z" in knobs:
        knobs["z"] = float(knobs["z"])
    prompt = read(pr.get("prompt_file", "proposer_prompt.md"))
    if not prompt.strip():
        raise StrategyError("the proposer prompt is empty")
    if len(prompt) > PROMPT_MAX:
        raise StrategyError(f"the proposer prompt has {len(prompt):,} characters; at most {PROMPT_MAX:,}")
    preset = data.get("based_on", default_preset)
    s = Strategy(name=str(data.get("name", preset)), preset=preset, description=str(data.get("description", "")), prompt=prompt, **knobs)
    est = estimate(s, full=True)
    if est["usd"] > USD_CAP:
        raise StrategyError(f"worst case ${est['usd']:.2f} of HF credits is over the ${USD_CAP:.0f} cap per run: lower iterations, "
                            f"val_trials or candidates")
    return s


def load(spec: str | Path) -> Strategy:
    """A preset name, a strategy folder, or the path of its strategy.toml."""
    path = Path(spec)
    if path.is_file():
        path = path.parent
    if path.is_dir() and (path / "strategy.toml").is_file():
        s = _parse((path / "strategy.toml").read_text(), lambda f: (path / f).read_text(), DEFAULT)
        if s.preset not in presets():
            raise StrategyError(f"based_on = {s.preset!r}: name the preset you started from ({', '.join(presets())})")
        return s
    if str(spec) in presets():
        p = _root().joinpath(str(spec))
        return _parse(p.joinpath("strategy.toml").read_text(), lambda f: p.joinpath(f).read_text(), str(spec))
    raise StrategyError(f"no strategy {str(spec)!r}: use a preset ({', '.join(presets())}) or a folder made by `aidev-evolve strategy-init`")


def init(preset: str, dest: Path) -> Path:
    """Copy a preset into dest for editing (refuses an existing folder)."""
    load(preset)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists: pick a new folder")
    dest.mkdir(parents=True)
    src = _root().joinpath(preset)
    for name in ("strategy.toml", "proposer_prompt.md"):
        (dest / name).write_text(src.joinpath(name).read_text())
    return dest


def iteration_minutes(s: Strategy) -> float:
    return ITER_MIN_TWO if s.candidates > 1 else ITER_MIN


def estimate(s: Strategy, splits: dict[str, Any] | None = None, *, full: bool = False) -> dict[str, Any]:
    """Wall time, HF Jobs and credits of a run. The workshop run: one proposal and one Job (a strategy with two
    candidates: three Jobs, its winner is measured again). `full`: the long run, with one re-measure and as many
    iterations as fit the session; credits are its worst case (all iterations, each re-measured)."""
    if splits is None:
        from aidev_evolve import loop

        splits = loop.default_splits()
    n_s, n_v, n_h = len(splits["search"]), len(splits["validation"]), len(splits["heldout"])
    if not full:
        ho = 2 * n_h * WORKSHOP_HELDOUT_TRIALS
        if s.candidates == 1:
            trials, jobs = n_v * s.val_trials * (2 if s.confirm else 1) + n_v * s.val_trials + ho, 1
            minutes = WORKSHOP_PROPOSE_MIN + WORKSHOP_JOB_MIN
        else:
            trials, jobs = s.candidates * (n_s + n_v * s.val_trials) + n_v * s.val_trials + 2 * n_v * s.val_trials + ho, 3
            minutes = WORKSHOP_PROPOSE_MIN + ITER_MIN_TWO + REMEASURE_MIN + HELDOUT_MIN
        return {"usd": round(trials * USD_PER_TRIAL + jobs * USD_PER_JOB, 2), "trials": trials, "jobs": jobs, "minutes": round(minutes, 1),
                "iterations_fit": WORKSHOP_ITERATIONS}
    per_iter = s.candidates * (n_s + n_v * s.val_trials) + n_v * s.val_trials + (2 * n_v * s.val_trials if s.remeasure else 0)
    trials = (n_s + 2 * n_v) + s.iterations * per_iter + 2 * n_h * HELDOUT_TRIALS
    jobs = 2 + s.iterations * (2 if s.remeasure else 1)
    usd = trials * USD_PER_TRIAL + jobs * USD_PER_JOB
    room = SESSION_MINUTES - CALIB_MIN - HELDOUT_MIN - (REMEASURE_MIN if s.remeasure else 0.0)
    fit = min(s.iterations, max(0, math.floor(room / iteration_minutes(s))))
    minutes = CALIB_MIN + fit * iteration_minutes(s) + (REMEASURE_MIN if s.remeasure else 0.0) + HELDOUT_MIN
    return {"usd": round(usd, 2), "trials": trials, "jobs": jobs, "minutes": round(minutes, 1), "iterations_fit": fit}


def next_fits(s: Strategy, *, elapsed_s: float, iteration_s: list[float], session_s: float) -> bool:
    """Start another iteration only if it and the held-out check still end inside the session. The next iteration is
    estimated as the longest one so far (the caller passes iterations without a re-measure) or the prior."""
    next_s = max(iteration_s) if iteration_s else 60 * iteration_minutes(s)
    return elapsed_s + next_s + 60 * HELDOUT_MIN <= session_s


def with_knobs(s: Strategy, **kw: Any) -> Strategy:
    """A copy with other knobs (tests and devtools); validated like a file."""
    for k, v in kw.items():
        _check(k, v)
    return replace(s, **kw)


def describe(s: Strategy) -> str:
    est, full = estimate(s), estimate(s, full=True)
    k = s.knobs()
    return (f"{s.name}{' (edited ' + s.preset + ')' if s.modified else ''} [{s.hash}]: {s.description}\n"
            f"  evidence: {k['traces']} trajectories (max {k['max_traces']}), history {k['history']}{', with diffs' if k['diffs'] else ''}; "
            f"{k['candidates']} candidate(s) per iteration; keep: {k['val_trials']} validation trials, z = {k['z']:g}, "
            f"re-measure {'on' if s.remeasure else 'off'}; up to {k['iterations']} iterations with --full\n"
            f"  workshop run: 1 iteration, {est['jobs']} HF Job{'s' if est['jobs'] > 1 else ''}, about {est['minutes']:.0f} min when run alone, about ${est['usd']:.2f} of HF credits\n"
            f"  --full: about {full['minutes']:.0f} min ({full['iterations_fit']} iterations fit the {SESSION_MINUTES:.0f}-min time box), "
            f"at most ${full['usd']:.2f}")
