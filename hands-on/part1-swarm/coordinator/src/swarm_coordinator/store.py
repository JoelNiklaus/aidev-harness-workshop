"""The claim queue: SQLite in one process, every operation under one lock (atomic by construction).

Leases last 40 minutes (PROTOCOL.md); an expired lease puts its cell back at its place in the
order. Results and the active leases go to the sink (the private HF bucket on the Space), from
which a restarted coordinator rebuilds its state (`restore`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from .grid import Grid

LEASE_S = 40 * 60
MAX_N = 2

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

    def __init__(self, held: list[str]):
        super().__init__(f"you already hold {len(held)} active lease(s): {', '.join(held)}; release or finish them first")
        self.held = held


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
                 sink: Sink | None = None, db_path: str = ":memory:", lease_s: float = LEASE_S):
        self.grid, self.clock, self.sink, self.lease_s = grid, clock, sink, lease_s
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
        return [{"cell_id": c, "user": u, "expires": e} for c, u, e in self._db.execute(
            "SELECT cell_id, lease_user, lease_expires FROM cells WHERE done=0 AND lease_user IS NOT NULL AND lease_expires>? ORDER BY ord", (now,))]

    def _changed(self, now: float) -> None:
        self.version += 1
        if self.sink is not None:
            self.sink.put_leases(self._leases(now))

    def _claim_doc(self, cid: str, h: str, t: str, k: int, expires: float) -> dict[str, Any]:
        return {"cell_id": cid, "harness": h, "task": t, "trial": k, "model": self.grid.model,
                "lease_expires_at": iso(expires), "run_config": dict(self.grid.run_config)}

    # ------------------------------------------------------------ protocol operations

    def claim(self, user: str, n: int = 1) -> list[dict[str, Any]]:
        if n not in range(1, MAX_N + 1):
            raise Invalid(f"n must be 1 or {MAX_N}")
        with self._lock:
            now = self.clock()
            held = [r[0] for r in self._db.execute(
                "SELECT cell_id FROM cells WHERE done=0 AND lease_user=? AND lease_expires>? ORDER BY ord", (user, now))]
            want = n - len(held)
            if want <= 0:
                raise AtLimit(held)
            rows = self._db.execute(
                "SELECT cell_id, harness, task, trial FROM cells WHERE done=0 AND (lease_user IS NULL OR lease_expires<=?) ORDER BY ord LIMIT ?",
                (now, want)).fetchall()
            if not rows:
                return []
            expires = now + self.lease_s
            self._db.executemany("UPDATE cells SET lease_user=?, lease_expires=?, n_claims=n_claims+1 WHERE cell_id=?",
                                 [(user, expires, r[0]) for r in rows])
            self._changed(now)
            return [self._claim_doc(*r, expires) for r in rows]

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
            return self._claim_doc(row[0], row[1], row[2], row[3], expires)

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
                raise Conflict(f"the lease on {cid} belongs to someone else")
            stored = {**result, "hf_user": user, "cost_usd_reported": result["cost_usd"],
                      "cost_usd": round(cost_usd(result, self.price), 6), "received_at": iso(now)}
            self._db.execute("INSERT INTO results(cell_id, hf_user, received, json) VALUES (?,?,?,?)", (cid, user, now, json.dumps(stored)))
            self._db.execute("UPDATE cells SET done=1, lease_user=NULL, lease_expires=NULL WHERE cell_id=?", (cid,))
            if self.sink is not None:
                self.sink.put(stored)
            self._changed(now)
            return stored

    # ------------------------------------------------------------ reads and restart

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
        """Rebuild results and live leases from the sink (after a Space restart). Returns the results loaded."""
        if self.sink is None:
            return 0
        results, leases = self.sink.load()
        with self._lock:
            now = self.clock()
            for r in results:
                self._db.execute("INSERT OR REPLACE INTO results(cell_id, hf_user, received, json) VALUES (?,?,?,?)",
                                 (r["cell_id"], r.get("hf_user"), _parse_ts(r.get("received_at"), now), json.dumps(r)))
                self._db.execute("UPDATE cells SET done=1 WHERE cell_id=?", (r["cell_id"],))
            for le in leases:
                if le["expires"] > now:
                    self._db.execute("UPDATE cells SET lease_user=?, lease_expires=?, n_claims=n_claims+1 WHERE cell_id=? AND done=0",
                                     (le["user"], le["expires"], le["cell_id"]))
            self.version += 1
        return len(results)


def _parse_ts(s: str | None, default: float) -> float:
    try:
        return datetime.fromisoformat(s).timestamp() if s else default
    except ValueError:
        return default
