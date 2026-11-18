"""Part 2 room board (/evolve): every attendee's evolve loop posts its events here (PROTOCOL.md, Part 2 section).

- `validate`: one event of `aidev-evolve/1`, bounded field by field; fields outside the protocol are dropped.
- `EvolveStore`: SQLite in one process, idempotent on (user, run_id, kind, iteration); each event also goes to
  the sink as `evolve/<user>/<run_id>/<kind>-<iteration>.json`, from which a restarted Space rebuilds it.
- `RateLimiter`: a token bucket per user (the client sends about one event per 4-5 minutes).
- `build_state`: the board payload: race curves, room best, seed baseline, feed of kept patches, fix tally, totals.
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


def one_line(hypothesis: str | None, max_len: int = 120) -> str:
    """The hypothesis's first sentence-line, without the 'Hypothesis:' label, capped for the projector."""
    line = (hypothesis or "").strip().splitlines()[0].strip() if (hypothesis or "").strip() else ""
    line = re.sub(r"^hypothesis:\s*", "", line, flags=re.I)  # not re-capitalised: it often starts with a task or file name
    return line if len(line) <= max_len else line[:max_len - 1].rstrip() + "…"


# ---- board payload ----------------------------------------------------------------------------------


def build_state(events: list[dict[str, Any]], reference: list[dict[str, Any]], *, mode: str = "live") -> dict[str, Any]:
    """`events`: EvolveStore.events() (arrival order). `reference`: recorded runs [{label, curve}] (live-mode backdrop)."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    feed: list[dict[str, Any]] = []
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in events:
        key = (e["user"], e["run_id"])
        r = runs.setdefault(key, {"run_id": e["run_id"], "user": e["user"], "points": {}, "kept": 0, "iterations": 0, "spend_usd": 0.0,
                                  "heldout": None, "last_at": 0.0})
        r["spend_usd"] = max(r["spend_usd"], e.get("spend_usd") or 0.0)  # spend only grows; the latest is the largest
        r["last_at"] = max(r["last_at"], e["received_at"])
        if e["kind"] == "seed":
            r["points"][0] = e["val_score"]
        elif e["kind"] == "iteration":
            r["points"][e["iteration"]] = e["best_val"]
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
        curve = [[x, r["points"][x]] for x in sorted(r["points"])]
        out_runs.append({"run_id": r["run_id"], "user": r["user"], "curve": curve, "best": max((y for _, y in curve), default=None),
                         "seed": r["points"].get(0), "kept": r["kept"], "iterations": r["iterations"], "spend_usd": round(r["spend_usd"], 4),
                         "heldout": r["heldout"], "last_at": r["last_at"]})

    max_x = max((c[0] for r in out_runs for c in r["curve"]), default=0)
    room_best = []
    for x in range(max_x + 1):
        ys = [y for r in out_runs for cx, y in r["curve"] if cx <= x]
        if ys:
            room_best.append([x, max(ys)])
    seeds = [r["seed"] for r in out_runs if r["seed"] is not None]
    if not seeds:
        seeds = [c[1] for ref in reference for c in ref["curve"] if c[0] == 0]
    baseline = sum(seeds) / len(seeds) if seeds else None

    totals = {"attendees": len({r["user"] for r in out_runs}), "runs": len(out_runs), "iterations": sum(r["iterations"] for r in out_runs),
              "kept": sum(r["kept"] for r in out_runs), "spend_usd": round(sum(r["spend_usd"] for r in out_runs), 4)}
    out_runs.sort(key=lambda r: (-(r["best"] if r["best"] is not None else -1), -r["last_at"]))
    fixes = sorted(({"label": k, "tried": v[0], "kept": v[1]} for k, v in tally.items()), key=lambda f: (-f["tried"], -f["kept"], f["label"]))
    feed.reverse()  # events come in arrival order: newest first
    return {"mode": mode, "totals": totals, "runs": out_runs[:MAX_RUNS], "room_best": room_best, "baseline": baseline,
            "reference": reference, "feed": feed[:FEED_LEN], "fixes": fixes[:FIXES_LEN], "max_iteration": max_x}
