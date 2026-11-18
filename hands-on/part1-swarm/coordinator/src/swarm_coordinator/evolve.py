"""Part 2 room board (/evolve): every attendee's evolve loop posts its events here (PROTOCOL.md, Part 2 section).

- `validate`: one event of `aidev-evolve/1`, bounded field by field; fields outside the protocol are dropped.
- `EvolveStore`: SQLite in one process, idempotent on (user, run_id, kind, iteration); each event also goes to
  the sink as `evolve/<user>/<run_id>/<kind>-<iteration>.json`, from which a restarted Space rebuilds it.
- `RateLimiter`: a token bucket per user (the client sends about one event per 4-5 minutes).
- `champion_curve`: a run's race line: the validation pass rate of its current harness, a step at each kept patch.
- `build_state`: the board payload: race curves, the leading run, seed baseline, feed of kept patches, fix tally, totals.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from . import ranking
from .store import Invalid

SCHEMA = "aidev-evolve/1"
MAX_BODY_BYTES = 64_000  # a full event with an 8,000-char diff is under 20 KB of JSON
MAX_RUNS = 200  # runs in the payload (best first); totals count all
MAX_RUNS_PER_USER = 10
FEED_LEN = 12
FIXES_LEN = 6
SINK_PREFIX = "evolve/"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

# ---- validation -------------------------------------------------------------------------------------


def _num(v: Any, name: str, lo: float, hi: float, *, null: bool = False) -> float | None:
    if v is None and null:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise Invalid(f"{name} must be a number in [{lo:g}, {hi:g}]")
    return float(v)


def _int(v: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise Invalid(f"{name} must be an integer in [{lo}, {hi}]")
    return v


def _str(v: Any, name: str, max_len: int, *, null: bool = False) -> str | None:
    """A string, cut to max_len: the client drops an event the board refuses, so verbose text is shortened, not refused."""
    if v is None and null:
        return None
    if not isinstance(v, str):
        raise Invalid(f"{name} must be a string")
    return v if len(v) <= max_len else v[:max_len - 1] + "…"


def _tasks(v: Any, name: str) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise Invalid(f"{name} must be a list of task names")
    return [_str(t, name, 100) for t in v[:100]]  # type: ignore[misc]


PRESET = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
HASH = re.compile(r"^[0-9a-f]{6,64}$")
KNOB_CHOICES = {"traces": ("failed", "all"), "history": ("none", "titles", "full")}
KNOB_RANGES = {"max_traces": (1, 8), "candidates": (1, 2), "val_trials": (1, 3), "iterations": (1, 6)}
PROMPT_MAX, MAX_PER_TASK, MAX_KEPT, MAX_SIBLINGS = 4000, 40, 20, 4


def _bool(v: Any, name: str) -> bool:
    if not isinstance(v, bool):
        raise Invalid(f"{name} must be true or false")
    return v


def _knobs(v: Any) -> dict[str, Any]:
    """The strategy's knobs (part2-evolve PROTOCOL.md): known keys only, each in its range."""
    if not isinstance(v, dict):
        raise Invalid("strategy.knobs must be an object")
    out: dict[str, Any] = {}
    for k, choices in KNOB_CHOICES.items():
        if k in v:
            if v[k] not in choices:
                raise Invalid(f"strategy.knobs.{k} must be one of {', '.join(choices)}")
            out[k] = v[k]
    for k, (lo, hi) in KNOB_RANGES.items():
        if k in v:
            out[k] = _int(v[k], f"strategy.knobs.{k}", lo, hi)
    if "z" in v:
        out["z"] = _num(v["z"], "strategy.knobs.z", 0, 3)
    for k in ("diffs", "confirm"):
        if k in v:
            out[k] = _bool(v[k], f"strategy.knobs.{k}")
    return out


def _strategy(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise Invalid("strategy must be an object")
    preset, h = v.get("preset"), v.get("hash")
    if not isinstance(preset, str) or not PRESET.match(preset):
        raise Invalid("strategy.preset must be a preset name")
    if not isinstance(h, str) or not HASH.match(h):
        raise Invalid("strategy.hash must be hex")
    return {"name": _str(v.get("name"), "strategy.name", 60), "preset": preset, "hash": h,
            "modified": _bool(v.get("modified", False), "strategy.modified"),
            "prompt": _str(v.get("prompt") or "", "strategy.prompt", PROMPT_MAX), "knobs": _knobs(v.get("knobs") or {})}


def _per_task(v: Any) -> dict[str, list[int]]:
    """{task: [seed passes, seed trials, final passes, final trials]}, at most MAX_PER_TASK tasks."""
    if not isinstance(v, dict):
        raise Invalid("per_task must be an object")
    out = {}
    for t, c in list(v.items())[:MAX_PER_TASK]:
        if not (isinstance(c, list) and len(c) == 4):
            raise Invalid("per_task values must be [seed_passes, seed_trials, final_passes, final_trials]")
        sp, sn, fp, fn = (_int(x, "per_task count", 0, 100) for x in c)
        if sp > sn or fp > fn:
            raise Invalid("per_task passes cannot exceed trials")
        out[_str(t, "per_task task", 100)] = [sp, sn, fp, fn]
    return out


def _siblings(v: Any) -> list[dict[str, Any]]:
    if not isinstance(v, list):
        raise Invalid("siblings must be a list")
    out = []
    for sib in v[:MAX_SIBLINGS]:
        if not isinstance(sib, dict):
            raise Invalid("each sibling must be an object")
        out.append({"candidate": _str(sib.get("candidate"), "sibling candidate", 20, null=True), "title": _str(sib.get("title"), "sibling title", 200, null=True),
                    "val": _num(sib.get("val"), "sibling val", 0, 1, null=True), "verdict": _str(sib.get("verdict"), "sibling verdict", 80, null=True)})
    return out


def validate(event: Any) -> dict[str, Any]:
    """The event as stored: known fields only, each checked, text cut to its display limit. Raises Invalid (422)."""
    if not isinstance(event, dict):
        raise Invalid("the event must be a JSON object")
    if event.get("schema") != SCHEMA:
        raise Invalid(f"schema must be {SCHEMA!r}")
    run_id = event.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise Invalid("run_id must be 1-100 characters of letters, digits, '.', '_' or '-'")
    kind = event.get("kind")
    out: dict[str, Any] = {"schema": SCHEMA, "run_id": run_id, "kind": kind, "spend_usd": _num(event.get("spend_usd", 0.0), "spend_usd", 0, 10_000)}
    if kind == "seed":
        out.update(iteration=0, val_score=_num(event.get("val_score"), "val_score", 0, 1),
                   search_score=_num(event.get("search_score"), "search_score", 0, 1, null=True))
    elif kind == "iteration":
        accepted = event.get("accepted")
        if not isinstance(accepted, bool):
            raise Invalid("accepted must be true or false")
        out.update(iteration=_int(event.get("iteration"), "iteration", 1, 500), candidate=_str(event.get("candidate"), "candidate", 20, null=True),
                   parent=_str(event.get("parent"), "parent", 20, null=True), title=_str(event.get("title"), "title", 200, null=True),
                   hypothesis=_str(event.get("hypothesis"), "hypothesis", 1000, null=True), accepted=accepted,
                   verdict=_str(event.get("verdict"), "verdict", 80, null=True),
                   val_child=_num(event.get("val_child"), "val_child", 0, 1, null=True), val_parent=_num(event.get("val_parent"), "val_parent", 0, 1, null=True),
                   best_val=_num(event.get("best_val"), "best_val", 0, 1), fixed=_tasks(event.get("fixed"), "fixed"), broke=_tasks(event.get("broke"), "broke"),
                   diff=_str(event.get("diff") or "", "diff", 8000))
    elif kind == "heldout":
        out.update(iteration=0, seed=_num(event.get("seed"), "seed", 0, 1), best=_num(event.get("best"), "best", 0, 1),
                   best_candidate=_str(event.get("best_candidate"), "best_candidate", 20, null=True),
                   n_tasks=_int(event.get("n_tasks"), "n_tasks", 0, 1000), trials=_int(event.get("trials"), "trials", 0, 100),
                   diff=_num(event.get("diff"), "diff", -1, 1, null=True), ci_low=_num(event.get("ci_low"), "ci_low", -1, 1, null=True),
                   ci_high=_num(event.get("ci_high"), "ci_high", -1, 1, null=True))
    else:
        raise Invalid("kind must be 'seed', 'iteration' or 'heldout'")
    # strategy fields (client 0.4.0, additive) and the provider that served the run (PROTOCOL.md "Providers")
    if event.get("strategy") is not None:
        out["strategy"] = _strategy(event["strategy"])
    if event.get("provider") is not None:
        out["provider"] = _str(event["provider"], "provider", 40)
    if kind == "iteration" and event.get("siblings"):
        out["siblings"] = _siblings(event["siblings"])
    if kind == "heldout":
        if event.get("per_task") is not None:
            out["per_task"] = _per_task(event["per_task"])
        if event.get("kept") is not None:
            if not isinstance(event["kept"], list):
                raise Invalid("kept must be a list of titles")
            out["kept"] = [_str(t, "kept title", 200) for t in event["kept"][:MAX_KEPT]]
    return out


# ---- store ------------------------------------------------------------------------------------------


def sink_path(user: str, run_id: str, kind: str, iteration: int) -> str:
    return f"{SINK_PREFIX}{user}/{run_id}/{kind}-{iteration:03d}.json"


class EvolveStore:
    def __init__(self, *, clock: Callable[[], float] = time.time, sink: Any = None, max_runs_per_user: int = MAX_RUNS_PER_USER):
        self.clock, self.sink, self.max_runs_per_user = clock, sink, max_runs_per_user
        self._lock = threading.Lock()
        self._db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self._db.executescript("""
            CREATE TABLE evolve_events(user TEXT, run_id TEXT, kind TEXT, iteration INTEGER, received_at REAL, json TEXT, diff TEXT,
                                       PRIMARY KEY (user, run_id, kind, iteration));
        """)
        self.version = 0

    def _insert(self, user: str, event: dict[str, Any], received_at: float) -> None:
        # The diff text lives in its own column: the board needs only its size, and parsing 500 runs' diffs
        # on every state rebuild cost more than the rest of the rebuild. A retried POST replaces the event
        # but keeps its first arrival time (the feed's order).
        diff = event.get("diff") if event["kind"] == "iteration" else None
        summary = {k: v for k, v in event.items() if not (k == "diff" and diff is not None)}
        if event["kind"] == "iteration" and "strategy" in summary:  # the prompt rides on every event; seed and held-out keep it
            summary["strategy"] = {k: v for k, v in summary["strategy"].items() if k != "prompt"}
        if diff is not None:
            summary["diff_lines"] = list(diff_size(diff))
        self._db.execute("""INSERT INTO evolve_events VALUES (?,?,?,?,?,?,?)
                            ON CONFLICT(user, run_id, kind, iteration) DO UPDATE SET json=excluded.json, diff=excluded.diff""",
                         (user, event["run_id"], event["kind"], event["iteration"], received_at, json.dumps(summary), diff))

    def add(self, user: str, event: dict[str, Any]) -> dict[str, Any]:
        """Store a validated event for `user`. Raises Invalid when the user already has too many runs."""
        with self._lock:
            runs = {r for (r,) in self._db.execute("SELECT DISTINCT run_id FROM evolve_events WHERE user=?", (user,))}
            if event["run_id"] not in runs and len(runs) >= self.max_runs_per_user:
                raise Invalid(f"at most {self.max_runs_per_user} runs per user on the board")
            now = self.clock()
            stored = {**event, "user": user, "received_at": now}
            self._insert(user, event, now)
            if self.sink is not None:
                self.sink.put_path(sink_path(user, event["run_id"], event["kind"], event["iteration"]),
                                   json.dumps(stored, separators=(",", ":")).encode())
            self.version += 1
            return stored

    def events(self) -> list[dict[str, Any]]:
        """Every event in arrival order, with `user` and `received_at`; an iteration's diff as `diff_lines` [added, removed]."""
        with self._lock:
            rows = self._db.execute("SELECT user, received_at, json FROM evolve_events ORDER BY received_at, rowid").fetchall()
        return [{**json.loads(j), "user": u, "received_at": t} for u, t, j in rows]

    def diff(self, user: str, run_id: str, iteration: int) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT diff FROM evolve_events WHERE user=? AND run_id=? AND kind='iteration' AND iteration=?",
                                   (user, run_id, iteration)).fetchone()
        return row[0] if row else None

    def restore(self) -> int:
        """Rebuild from the sink (after a Space restart). Returns the events loaded."""
        if self.sink is None:
            return 0
        files = self.sink.load_prefix(SINK_PREFIX)
        n = 0
        with self._lock:
            for data in files.values():
                e = json.loads(data)
                user, received = e.pop("user"), e.pop("received_at", self.clock())
                self._insert(user, e, received)
                n += 1
            self.version += 1
        return n


# ---- rate limit -------------------------------------------------------------------------------------


class RateLimiter:
    """Token bucket per key: `burst` events at once, then `per_minute`. `allow` returns 0 or the seconds to wait."""

    def __init__(self, per_minute: float, burst: int, *, clock: Callable[[], float] = time.monotonic):
        self.rate, self.burst, self.clock = per_minute / 60.0, burst, clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> float:
        with self._lock:
            now = self.clock()
            tokens, at = self._buckets.get(key, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - at) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1.0 - tokens) / self.rate


# ---- fix tally --------------------------------------------------------------------------------------

# First match wins, title before hypothesis: each proposal lands in exactly one row, so the rows add up.
FIX_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("Time budget, not a step limit", re.compile(r"\bsteps?\b|max_steps|time[- ]based|time budget|until time|step limit|step cap")),
    ("Show the clock", re.compile(r"clock|remaining time|time left|deadline|wrap-up")),
    ("Show stderr", re.compile(r"stderr")),
    ("Longer or smarter output", re.compile(r"output|truncat|head and tail")),
    ("Command timeouts", re.compile(r"timeout|\bhang|\bhung")),
    ("Verify before finishing", re.compile(r"verif|before (finishing|declaring|done)|acceptance test|grader|review|checklist|requirement")),
    ("Break loops and stalls", re.compile(r"loop|stall|repeat|switch approach|checkpoint|progress")),
    ("Keep shell state", re.compile(r"\bcwd\b|stateless|persistent shell|working dir|environment variables")),
    ("Work routine in the prompt", re.compile(r"prompt|routine|habit|\bplan|method|docs")),
]
OTHER = "Other"


def fix_category(title: str | None, hypothesis: str | None) -> str:
    for text in (title or "", hypothesis or ""):
        low = text.lower()
        for label, pat in FIX_CATEGORIES:
            if pat.search(low):
                return label
    return OTHER


def diff_size(diff: str | None) -> tuple[int, int]:
    """Lines added and removed in a unified diff (file headers not counted)."""
    added = removed = 0
    for line in (diff or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def one_line(hypothesis: str | None) -> str:
    """The hypothesis's first line, without the 'Hypothesis:' label. Not shortened: the page fits it to two
    lines at a sentence or clause boundary (`hypCandidates`), which needs the whole line."""
    line = (hypothesis or "").strip().splitlines()[0].strip() if (hypothesis or "").strip() else ""
    return re.sub(r"^hypothesis:\s*", "", line, flags=re.I)  # not re-capitalised: it often starts with a task or file name


# ---- race curves ------------------------------------------------------------------------------------


def steps(levels: list[tuple[int, float]]) -> list[list[float]]:
    """A step line through (x, level) points sorted by x: flat to the next x, then vertical to its level."""
    out: list[list[float]] = []
    for x, v in levels:
        if out and out[-1][0] != x:  # flat to x: stretch the last flat run instead of adding a point on it
            if len(out) >= 2 and out[-2][1] == out[-1][1] and out[-2][0] != out[-1][0]:
                out[-1][0] = x
            else:
                out.append([x, out[-1][1]])
        if not out or out[-1][1] != v:
            out.append([x, v])
    return out


def champion_curve(events: list[dict[str, Any]]) -> list[list[float]]:
    """One run's race line from its seed and iteration events (any order): the validation pass rate of the
    run's current harness. `best_val` is that harness's pooled score, so it moves when a dropped patch's
    iteration re-measures it; the line shows each harness at its latest score and steps only at a kept patch."""
    pts = sorted((0, e["val_score"], True) if e["kind"] == "seed" else (e["iteration"], e["best_val"], e["accepted"])
                 for e in events if e["kind"] in ("seed", "iteration"))
    segments: list[list[Any]] = []  # [start x, last x, latest score]
    for x, v, kept in pts:
        if kept or not segments:
            segments.append([x, x, v])
        else:
            segments[-1][1:] = [x, v]
    levels = [(x, seg[2]) for seg in segments for x in (seg[0], seg[1])]
    return steps(levels)


def level_at(curve: list[list[float]], x: int) -> float | None:
    """The curve's level just after x (after a step at x), None before the curve starts."""
    v = None
    for cx, cy in curve:
        if cx > x:
            break
        v = cy
    return v


# ---- strategy ranking -------------------------------------------------------------------------------


def ranking_runs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The runs `ranking.rank` takes: every run whose held-out event carries per-task counts and a strategy."""
    out = []
    for e in events:
        if e["kind"] == "heldout" and e.get("per_task") and e.get("strategy"):
            out.append({"run_id": e["run_id"], "user": e["user"], "strategy": e["strategy"],
                        "heldout": {"per_task": e["per_task"], "best_candidate": e.get("best_candidate")}, "kept": e.get("kept") or []})
    return out


def _rank_both(runs: list[dict[str, Any]], rank: Callable[..., dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    return rank(runs, by="hash"), rank(runs, by="preset")


class RankCache:
    """`ranking.rank` takes ~2 s for 500 runs (task bootstrap) and a held-out result arrives once per run, at its end.
    Rank again only when the held-out results changed, and then only after 10x the last ranking's cost (cheap while
    few runs have ended, at most every ~30 s at 500): in between the last ranking is served."""

    COST_FACTOR = 10.0

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, rank: Callable[..., dict[str, Any]] = ranking.rank):
        self.clock, self.rank = clock, rank
        self.key: Any = None
        self.at = self.cost = 0.0
        self.value: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def get(self, runs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        key = tuple(sorted((r["user"], r["run_id"], json.dumps(r["heldout"]["per_task"], sort_keys=True), r["strategy"]["hash"]) for r in runs))
        with self._lock:
            now = self.clock()
            if self.value is None or (key != self.key and now - self.at >= self.COST_FACTOR * self.cost):
                self.value, self.key = _rank_both(runs, self.rank), key
                self.at = self.clock()
                self.cost = self.at - now
            return self.value


def _with_details(result: dict[str, Any], events: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The ranking plus what the Winner view shows: each winning strategy's knobs and prompt (a preset pooled with
    edited copies shows the preset's own), and the winning patch's kept iterations with their diff links."""
    w = result.get("winner")
    if not w:
        return result
    by = result["by"]
    strategies = []
    for k in w["keys"]:
        members = [r["strategy"] for r in runs if r["strategy"].get(by, r["strategy"]["hash"]) == k]
        st = next((m for m in members if not m.get("modified")), members[0])
        entry = next(e for e in result["strategies"] if e["key"] == k)
        strategies.append({"key": k, "name": entry["name"], "preset": st.get("preset"), "knobs": st.get("knobs") or {}, "prompt": st.get("prompt") or "",
                           "n_runs": entry["n_runs"], "gain": entry["gain"], "ci": entry["ci"]})
    p = dict(w["patch"])
    p["iterations"] = [{"iteration": e["iteration"], "title": e.get("title") or "(untitled patch)",
                        "diff_url": f"api/evolve/diff/{e['user']}/{e['run_id']}/{e['iteration']}"}
                       for e in sorted((e for e in events if e["kind"] == "iteration" and e["accepted"] and e["user"] == p["user"]
                                        and e["run_id"] == p["run_id"]), key=lambda e: e["iteration"])]
    return {**result, "winner": {**w, "strategies": strategies, "patch": p}}


# ---- board payload ----------------------------------------------------------------------------------


def build_state(events: list[dict[str, Any]], reference: list[dict[str, Any]], *, mode: str = "live",
                rank_cache: RankCache | None = None) -> dict[str, Any]:
    """`events`: EvolveStore.events() (arrival order). `reference`: recorded runs [{label, curve}] (live-mode backdrop).
    `rank_cache`: reuse the strategy ranking between rebuilds (None: rank now)."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    feed: list[dict[str, Any]] = []
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in events:
        key = (e["user"], e["run_id"])
        r = runs.setdefault(key, {"run_id": e["run_id"], "user": e["user"], "events": [], "kept": 0, "iterations": 0, "spend_usd": 0.0,
                                  "heldout": None, "last_at": 0.0, "strategy": None, "preset": None, "provider": None})
        if e.get("strategy"):
            r["strategy"], r["preset"] = e["strategy"].get("name"), e["strategy"].get("preset")
        r["provider"] = e.get("provider") or r["provider"]
        r["spend_usd"] = max(r["spend_usd"], e.get("spend_usd") or 0.0)  # spend only grows; the latest is the largest
        r["last_at"] = max(r["last_at"], e["received_at"])
        if e["kind"] == "seed":
            r["events"].append(e)
        elif e["kind"] == "iteration":
            r["events"].append(e)
            r["iterations"] += 1
            r["kept"] += e["accepted"]
            row = tally[fix_category(e.get("title"), e.get("hypothesis"))]
            row[0] += 1
            row[1] += e["accepted"]
            if e["accepted"]:
                added, removed = e["diff_lines"]
                delta = (e["val_child"] - e["val_parent"]) if e.get("val_child") is not None and e.get("val_parent") is not None else None
                feed.append({"user": e["user"], "run_id": e["run_id"], "iteration": e["iteration"], "title": e.get("title") or "(untitled patch)",
                             "hypothesis": one_line(e.get("hypothesis")), "delta": delta, "added": added, "removed": removed,
                             "n_fixed": len(e.get("fixed") or []), "at": e["received_at"]})
        elif e["kind"] == "heldout":
            r["heldout"] = {k: e.get(k) for k in ("seed", "best", "n_tasks", "trials", "diff", "ci_low", "ci_high")}

    out_runs = []
    for r in runs.values():
        curve = champion_curve(r["events"])
        seed = next((e["val_score"] for e in r["events"] if e["kind"] == "seed"), None)
        out_runs.append({"run_id": r["run_id"], "user": r["user"], "curve": curve, "current": curve[-1][1] if curve else None,
                         "seed": seed, "kept": r["kept"], "iterations": r["iterations"], "spend_usd": round(r["spend_usd"], 4),
                         "heldout": r["heldout"], "last_at": r["last_at"], "strategy": r["strategy"], "preset": r["preset"], "provider": r["provider"]})

    max_x = max((c[0] for r in out_runs for c in r["curve"]), default=0)
    levels = []
    for x in range(max_x + 1):
        ys = [v for r in out_runs if (v := level_at(r["curve"], x)) is not None]
        if ys:
            levels.append((x, max(ys)))
    leader = steps(levels)  # the highest current harness at each iteration
    seeds = [r["seed"] for r in out_runs if r["seed"] is not None]
    if not seeds:
        seeds = [c[1] for ref in reference for c in ref["curve"][:1] if c[0] == 0]
    baseline = sum(seeds) / len(seeds) if seeds else None

    totals = {"attendees": len({r["user"] for r in out_runs}), "runs": len(out_runs), "iterations": sum(r["iterations"] for r in out_runs),
              "kept": sum(r["kept"] for r in out_runs), "spend_usd": round(sum(r["spend_usd"] for r in out_runs), 4)}
    out_runs.sort(key=lambda r: (-(r["current"] if r["current"] is not None else -1), -r["last_at"]))
    fixes = sorted(({"label": k, "tried": v[0], "kept": v[1]} for k, v in tally.items()), key=lambda f: (-f["tried"], -f["kept"], f["label"]))
    feed.reverse()  # events come in arrival order: newest first
    ranked = ranking_runs(events)
    by_hash, by_preset = rank_cache.get(ranked) if rank_cache is not None else _rank_both(ranked, ranking.rank)
    return {"mode": mode, "totals": totals, "runs": out_runs[:MAX_RUNS], "leader": leader, "baseline": baseline,
            "reference": reference, "feed": feed[:FEED_LEN], "fixes": fixes[:FIXES_LEN], "max_iteration": max_x,
            "strategies": _with_details(by_hash, events, ranked), "presets": _with_details(by_preset, events, ranked), "min_runs": ranking.MIN_RUNS}
