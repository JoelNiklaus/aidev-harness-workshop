"""The claim queue: SQLite in one process, every operation under one lock (atomic by construction).

Leases are short (LEASE_S, PROTOCOL.md) and the runner renews its lease with a heartbeat every 30 s,
so the cell of a runner that dies without releasing it (a killed Job) returns to the queue about a
minute and a half after its last heartbeat. Expiry is lazy: every claim and every state read treats
a lease past its expiry as free, so no sweeper runs. Results and the active leases go to the sink
(the private HF bucket on the Space), from which a restarted coordinator rebuilds its state (`restore`).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from .grid import Grid

log = logging.getLogger("swarm.store")

LEASE_S = 90  # > 2 heartbeats of the runner (cli.HEARTBEAT_S = 30): one lost beat never costs a live trial its cell
# Leases one attendee may hold: a Job runs up to this many trials at once (aidev-swarm --parallel), each slot with its own
# heartbeats and one cell per claim. A runner asks for its slot count (`limit`, one more while it skips a failed cell, so
# the cell it gives back is not handed straight back to it); the share below caps it further.
MAX_LEASES = 9
# An attendee counts as active for the share while they hold a lease or claimed / heartbeated within this window.
ACTIVE_S = 3 * LEASE_S
# Slow start: an attendee's share starts at 2 (a runner's spare lease for skipping a failed cell, or a second Job) and
# grows by one lease every RAMP_S from their first claim, so the first arrivals of a room that launches within a few
# minutes do not take every cell before the others are seen (8 slots after 120 s; the fill-time cost with 40 attendees
# is about a minute, runner/PROGRESS.md).
RAMP_S = 20

RESULT_FIELDS = {
    "cell_id": str, "harness": str, "harness_version": str, "task": str, "trial": int, "model": str,
    "reward": (int, float), "passed": bool, "started_at": str, "finished_at": str, "wall_s": (int, float),
    "n_input_tokens": int, "n_cached_tokens": (int, type(None)), "n_output_tokens": int, "n_calls": (int, type(None)),
    "cost_usd": (int, float), "error": (str, type(None)), "hf_user": str, "runner_version": str,
    "sandbox_id": (str, type(None)), "trajectory_url": (str, type(None)),
}


class Conflict(Exception):
    """409: the cell's lease or result belongs to someone else."""


class AtLimit(Conflict):
    """409: the user already holds as many active leases as allowed."""

    def __init__(self, held: list[str], share: int | None = None):
        super().__init__(f"you already hold {len(held)} active lease(s): {', '.join(held)}; release or finish them first")
        self.held, self.share = held, share


class Invalid(ValueError):
    """422: the request does not fit the protocol or the cell."""


class NotFound(KeyError):
    """404: no such cell in the grid."""


class Sink(Protocol):
    def put(self, result: dict[str, Any]) -> None: ...
    def put_leases(self, leases: list[dict[str, Any]]) -> None: ...
    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")


def cost_usd(result: dict[str, Any], price: dict[str, float]) -> float:
    """Tokens x list price (USD per 1M; cached input at its own rate), the board's rule."""
    inp = int(result.get("n_input_tokens") or 0)
    cached = min(int(result.get("n_cached_tokens") or 0), inp)
    return ((inp - cached) * price["input"] + cached * price["cached_input"] + int(result.get("n_output_tokens") or 0) * price["output"]) / 1e6


def validate(result: Any) -> None:
    if not isinstance(result, dict):
        raise Invalid("result must be a JSON object")
    for k, typ in RESULT_FIELDS.items():
        if k not in result:
            raise Invalid(f"result misses field {k!r}")
        v = result[k]
        if not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
            raise Invalid(f"result field {k!r} has the wrong type")


class Store:
    def __init__(self, grid: Grid, prices: dict[str, dict[str, Any]], *, clock: Callable[[], float] = time.time,
                 sink: Sink | None = None, db_path: str = ":memory:", lease_s: float = LEASE_S, extra_seconds: float = 0):
        self.grid, self.clock, self.sink, self.lease_s = grid, clock, sink, lease_s
        self.extra_seconds = extra_seconds
        self.start: float | None = None
        self.price = prices[grid.model]
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
            DROP TABLE IF EXISTS cells; DROP TABLE IF EXISTS results;
            CREATE TABLE cells(cell_id TEXT PRIMARY KEY, ord INTEGER UNIQUE, harness TEXT, task TEXT, trial INTEGER,
                               lease_user TEXT, lease_expires REAL, n_claims INTEGER DEFAULT 0, done INTEGER DEFAULT 0);
            CREATE INDEX cells_open ON cells(done, ord);
            CREATE TABLE results(cell_id TEXT PRIMARY KEY, hf_user TEXT, received REAL, json TEXT);
        """)
        self._db.executemany("INSERT INTO cells(cell_id, ord, harness, task, trial) VALUES (?,?,?,?,?)",
                             [(c.cell_id, i, c.harness, c.task, c.trial) for i, c in enumerate(grid.cells)])
        self.version = 0  # bumps on every change; the state cache keys on it
        self._seen: dict[str, float] = {}  # user -> last claim or heartbeat (in memory: a restart forgets it for ACTIVE_S)
        self._first: dict[str, float] = {}  # user -> first claim (slow start; a restart starts it again)

    # ------------------------------------------------------------ helpers (call under the lock)

    def _cell(self, cid: str) -> Any:
        row = self._db.execute("SELECT cell_id, harness, task, trial, lease_user, lease_expires, done FROM cells WHERE cell_id=?", (cid,)).fetchone()
        if row is None:
            raise NotFound(cid)
        return row

    @staticmethod
    def _active(row: Any, now: float) -> str | None:
        """The user holding a live lease on the row, else None."""
        return row[4] if row[4] is not None and row[5] is not None and row[5] > now else None

    def _leases(self, now: float) -> list[dict[str, Any]]:
        return [{"cell_id": c, "user": u, "expires": e, **({"room_start": self.start} if self.extra_seconds else {})} for c, u, e in self._db.execute(
            "SELECT cell_id, lease_user, lease_expires FROM cells WHERE done=0 AND lease_user IS NOT NULL AND lease_expires>? ORDER BY ord", (now,))]

    def _changed(self, now: float) -> None:
        self.version += 1
        if self.sink is not None:
            self.sink.put_leases(self._leases(now))

    def _claim_doc(self, cid: str, h: str, t: str, k: int, expires: float, share: int) -> dict[str, Any]:
        return {"cell_id": cid, "harness": h, "task": t, "trial": k, "model": self.grid.model,
                "lease_expires_at": iso(expires), "run_config": dict(self.grid.run_config), "share": share}

    def _share(self, user: str, now: float) -> int:
        """Leases `user` may hold now: the cells without a result spread over the active attendees, and at most one more
        per RAMP_S since their first claim, so early attendees with parallel runners do not take every cell while
        later ones arrive (at least 1)."""
        active = {u for u, t in self._seen.items() if t > now - ACTIVE_S}
        active.update(le["user"] for le in self._leases(now))
        open_cells = self._db.execute("SELECT COUNT(*) FROM cells WHERE done=0").fetchone()[0]
        ramp = 2 + int((now - self._first.get(user, now)) / RAMP_S)
        return max(1, min(ramp, math.ceil(open_cells / max(1, len(active)))))

    def _extra_cell(self, h: str, t: str, k: int) -> str:
        cid = f"{h}__{t}__{k}"
        self._db.execute("INSERT OR IGNORE INTO cells(cell_id,ord,harness,task,trial) SELECT ?,MAX(ord)+1,?,?,? FROM cells", (cid,h,t,k))
        return cid

    def _restore_extra(self, cid: str) -> None:
        parts = cid.split("__")
        if len(parts) == 3 and parts[0] in self.grid.harnesses and parts[1] in self.grid.tasks and parts[2].isdigit() and int(parts[2]) >= self.grid.trials_per_cell:
            self._extra_cell(parts[0], parts[1], int(parts[2]))

    # ------------------------------------------------------------ protocol operations

    def claim(self, user: str, limit: int = 1) -> list[dict[str, Any]]:
        """The next free cell (a list of one, empty when none is free) while `user` holds fewer than min(`limit`, share)
        live leases, else AtLimit."""
        if limit not in range(1, MAX_LEASES + 1):
            raise Invalid(f"limit must be from 1 to {MAX_LEASES}")
        with self._lock:
            now = self.clock()
            if self.start is None:
                self.start = now
            if self.extra_seconds and now >= self.start + self.extra_seconds:
                return []
            self._seen[user] = now
            self._first.setdefault(user, now)
            share = self._share(user, now)
            held = [r[0] for r in self._db.execute(
                "SELECT cell_id FROM cells WHERE done=0 AND lease_user=? AND lease_expires>? ORDER BY ord", (user, now))]
            if len(held) >= min(limit, share):
                raise AtLimit(held, share)
            rows = self._db.execute(
                "SELECT cell_id, harness, task, trial FROM cells WHERE done=0 AND (lease_user IS NULL OR lease_expires<=?) ORDER BY ord LIMIT 1",
                (now,)).fetchall()
            if not rows and self.extra_seconds:
                # Spread repeats across pairs; within each wave, shorter observed tasks go first.
                pair = self._db.execute("""SELECT harness,task,MAX(trial)+1 AS k FROM cells LEFT JOIN results USING(cell_id)
                    GROUP BY harness,task ORDER BY k,COALESCE(AVG(json_extract(json,'$.wall_s')),1e9),MIN(ord) LIMIT 1""").fetchone()
                if pair:
                    h,t,k = pair
                    rows = [(self._extra_cell(h,t,k),h,t,k)]
            if not rows:
                return []
            expires = now + self.lease_s
            self._db.executemany("UPDATE cells SET lease_user=?, lease_expires=?, n_claims=n_claims+1 WHERE cell_id=?",
                                 [(user, expires, r[0]) for r in rows])
            self._changed(now)
            return [self._claim_doc(*r, expires, share) for r in rows]

    def lease(self, cid: str, user: str) -> dict[str, Any]:
        """Claim one named cell (the replay's virtual runners use this to follow a fixed schedule)."""
        with self._lock:
            now = self.clock()
            row = self._cell(cid)
            holder = self._active(row, now)
            if row[6] or (holder and holder != user):
                raise Conflict(f"{cid} is taken")
            expires = now + self.lease_s
            self._db.execute("UPDATE cells SET lease_user=?, lease_expires=?, n_claims=n_claims+1 WHERE cell_id=?", (user, expires, cid))
            self._changed(now)
            return self._claim_doc(row[0], row[1], row[2], row[3], expires, self._share(user, now))

    def heartbeat(self, cid: str, user: str) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            row = self._cell(cid)
            holder = self._active(row, now)
            if row[6]:
                raise Conflict(f"{cid} already has a result")
            # a live lease of this user, or an expired one of this user that nobody took over
            if holder != user and (holder is not None or row[4] != user):
                raise Conflict(f"the lease on {cid} belongs to someone else")
            expires = now + self.lease_s
            self._seen[user] = now
            self._db.execute("UPDATE cells SET lease_user=?, lease_expires=? WHERE cell_id=?", (user, expires, cid))
            self._changed(now)
            return {"cell_id": cid, "lease_expires_at": iso(expires)}

    def release(self, cid: str, user: str) -> None:
        with self._lock:
            now = self.clock()
            row = self._cell(cid)
            holder = self._active(row, now)
            if holder is not None and holder != user:
                raise Conflict(f"the lease on {cid} belongs to someone else")
            if holder == user:
                self._db.execute("UPDATE cells SET lease_user=NULL, lease_expires=NULL WHERE cell_id=?", (cid,))
                self._changed(now)

    def submit(self, cid: str, user: str, result: Any) -> dict[str, Any]:
        validate(result)
        with self._lock:
            now = self.clock()
            row = self._cell(cid)
            if (result["cell_id"], result["harness"], result["task"], result["trial"], result["model"]) != (cid, row[1], row[2], row[3], self.grid.model):
                raise Invalid(f"result does not match cell {cid} ({row[1]}, {row[2]}, trial {row[3]}, {self.grid.model})")
            if row[6]:
                have = self._db.execute("SELECT hf_user, json FROM results WHERE cell_id=?", (cid,)).fetchone()
                if have and have[0] == user:
                    return json.loads(have[1])  # a retried submit: idempotent
                raise Conflict(f"{cid} already has a result")
            holder = self._active(row, now)
            if holder is not None and holder != user:
                # typically a late result: the sender's lease expired and the cell went to someone else meanwhile
                log.warning("late result for %s from %s rejected: %s holds the cell now", cid, user, holder)
                raise Conflict(f"the lease on {cid} belongs to someone else")
            stored = {**result, "hf_user": user, "cost_usd_reported": result["cost_usd"],
                      "cost_usd": round(cost_usd(result, self.price), 6), "received_at": iso(now), **({"room_start": self.start} if self.extra_seconds else {})}
            self._db.execute("INSERT INTO results(cell_id, hf_user, received, json) VALUES (?,?,?,?)", (cid, user, now, json.dumps(stored)))
            self._db.execute("UPDATE cells SET done=1, lease_user=NULL, lease_expires=NULL WHERE cell_id=?", (cid,))
            if self.sink is not None:
                self.sink.put(stored)
            self._changed(now)
            return stored

    # ------------------------------------------------------------ reads and restart

    def leased_open(self) -> int:
        """Cells without a result under a live lease: they come back if their runner dies."""
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM cells WHERE done=0 AND lease_user IS NOT NULL AND lease_expires>?",
                                    (self.clock(),)).fetchone()[0]

    def results(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(j) for (j,) in self._db.execute("SELECT json FROM results ORDER BY received, cell_id")]

    def snapshot(self) -> dict[str, Any]:
        """Everything the board needs, read in one consistent pass."""
        with self._lock:
            now = self.clock()
            leases = self._leases(now)
            results = [json.loads(j) for (j,) in self._db.execute("SELECT json FROM results ORDER BY received, cell_id")]
            max_claims = self._db.execute("SELECT COALESCE(MAX(n_claims), 0) FROM cells").fetchone()[0]
            return {"now": now, "leases": leases, "results": results, "max_claims": max_claims, "version": self.version}

    def restore(self) -> int:
        """Rebuild results and the saved leases from the sink (after a Space restart). Returns the results loaded."""
        if self.sink is None:
            return 0
        results, leases = self.sink.load()
        with self._lock:
            now = self.clock()
            if self.extra_seconds and (results or leases):
                times = [r["room_start"] for r in results + leases if isinstance(r.get("room_start"), (int, float))]
                self.start = min([now, *times])
            for r in results:
                self._restore_extra(r["cell_id"])
                self._db.execute("INSERT OR REPLACE INTO results(cell_id, hf_user, received, json) VALUES (?,?,?,?)",
                                 (r["cell_id"], r.get("hf_user"), _parse_ts(r.get("received_at"), now), json.dumps(r)))
                self._db.execute("UPDATE cells SET done=1 WHERE cell_id=?", (r["cell_id"],))
            # A restart outlasts a short lease: every saved lease starts a fresh one now, so a live runner's next
            # heartbeat keeps its cell and a dead runner's cell returns one lease later.
            for le in leases:
                self._restore_extra(le["cell_id"])
                self._db.execute("UPDATE cells SET lease_user=?, lease_expires=?, n_claims=n_claims+1 WHERE cell_id=? AND done=0",
                                 (le["user"], max(le["expires"], now + self.lease_s), le["cell_id"]))
            self.version += 1
        return len(results)


def _parse_ts(s: str | None, default: float) -> float:
    try:
        return datetime.fromisoformat(s).timestamp() if s else default
    except ValueError:
        return default
