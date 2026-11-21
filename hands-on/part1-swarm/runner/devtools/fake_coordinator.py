"""A protocol-faithful stand-in for the coordinator (PROTOCOL.md), stdlib only, for runner dry runs.

    python devtools/fake_coordinator.py --port 8765 [--grid ../config/grid.json] [--harnesses pi opencode] [--tasks t1 t2] [--trials 1]
                                        [--host 0.0.0.0] [--max-leases 25]

Hands out cells (ids `<harness>__<task>__<trial>`, as the real coordinator) one harness after the next, breadth-first over trials within a harness, 90-second
leases (renewed by heartbeats, as the real coordinator), and keeps results in memory (GET /api/state shows them, with the number of distinct client IPs).
Lease limit per user as in the real coordinator (`?limit=` caps the user's live leases, at most 9; one cell per claim;
no share of the room); `--max-leases`
raises it, so one account can run a many-Job dry run (the real coordinator has no such switch). Tokens are checked for presence only
(no whoami call) and never stored: the username is "user-<n>" per distinct token.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LEASE_S = 90  # the real coordinator's store.LEASE_S
MAX_LEASES = 9  # the real coordinator's store.MAX_LEASES


def cells(grid: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for h in grid["harnesses"]:
        for trial in range(grid["trials_per_cell"]):
            for t in grid["tasks"]:
                out.append({"cell_id": f"{h}__{t}__{trial}", "harness": h, "task": t, "trial": trial, "model": grid["model"]})
    return out


class State:
    def __init__(self, grid: dict[str, Any], lease_s: float = LEASE_S, max_leases: int | None = None, behind_jobs_proxy: bool = False):
        self.lock = threading.Lock()
        self.behind_jobs_proxy = behind_jobs_proxy
        self.max_leases = max_leases
        self.clients: set[str] = set()
        self.queue = cells(grid)
        self.leases: dict[str, tuple[str, float, dict[str, Any]]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.users: dict[int, str] = {}
        self.lease_s = lease_s
        self.run_config = {k: grid[k] for k in ("reasoning_effort", "provider", "benchmark") if k in grid}

    def user(self, token: str) -> str:
        return self.users.setdefault(hash(token), f"user-{len(self.users) + 1}")

    def _expire(self) -> None:
        now = time.time()
        for cid, (_, until, cell) in list(self.leases.items()):
            if until < now:
                del self.leases[cid]
                self.queue.insert(0, cell)

    def claim(self, user: str, limit: int) -> list[dict[str, Any]] | None:
        """The next cell (a list of one, empty when none is left); None when the user is at the lease limit (the real
        coordinator's 409 with `held`)."""
        with self.lock:
            self._expire()
            held = sum(1 for u, _, _ in self.leases.values() if u == user)
            if held >= (self.max_leases or limit):
                return None
            out = []
            while self.queue and not out:
                cell = self.queue.pop(0)
                until = time.time() + self.lease_s
                self.leases[cell["cell_id"]] = (user, until, cell)
                out.append({**cell, "lease_expires_at": datetime.fromtimestamp(until, UTC).isoformat(), "run_config": self.run_config})
            return out

    def owned(self, cid: str, user: str) -> bool:
        lease = self.leases.get(cid)
        return lease is not None and lease[0] == user

    def heartbeat(self, cid: str, user: str) -> bool:
        with self.lock:
            if not self.owned(cid, user):
                return False
            u, _, cell = self.leases[cid]
            self.leases[cid] = (u, time.time() + self.lease_s, cell)
            return True

    def submit(self, cid: str, user: str, result: dict[str, Any]) -> bool:
        with self.lock:
            done = self.results.get(cid)
            if done is not None:  # a retried submit by the same user is idempotent, as in the real coordinator
                return done["hf_user_seen"] == user
            if not self.owned(cid, user):
                return False
            del self.leases[cid]
            self.results[cid] = {**result, "hf_user_seen": user}
            return True

    def release(self, cid: str, user: str) -> bool:
        with self.lock:
            if not self.owned(cid, user):
                return False
            _, _, cell = self.leases.pop(cid)
            self.queue.insert(0, cell)
            return True

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {"open": len(self.queue), "leased": len(self.leases), "done": len(self.results), "client_ips": len(self.clients),
                    "results": list(self.results.values())}


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet; never logs headers (tokens)
            pass

        def _send(self, code: int, body: Any = None) -> None:
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                return self._send(200, state.state())
            return self._send(404, {"detail": "not found"})

        def do_POST(self):
            auth = self.headers.get("Authorization", "")
            if state.behind_jobs_proxy:  # the HF Jobs proxy checked the token (read access to the namespace) and consumed it
                user = "jobs-proxy-user"
            elif not auth.startswith("Bearer ") or len(auth) < 12:
                return self._send(401, {"detail": "missing token"})
            else:
                user = state.user(auth[7:])
            with state.lock:
                state.clients.add(self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip())
            url = urllib.parse.urlparse(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if url.path == "/api/claim":
                limit = min(int(urllib.parse.parse_qs(url.query).get("limit", ["1"])[0]), MAX_LEASES)
                got = state.claim(user, limit)
                if got is None:
                    held = sorted(c for c, (u, _, _) in state.leases.items() if u == user)
                    return self._send(409, {"detail": f"you already hold {len(held)} active lease(s)", "held": held,
                                            "share": state.max_leases or limit})
                if not got:
                    return self._send(204)
                return self._send(200, got[0])
            for verb in ("heartbeat", "submit", "release"):
                prefix = f"/api/{verb}/"
                if url.path.startswith(prefix):
                    cid = urllib.parse.unquote(url.path[len(prefix):])
                    args = (cid, user, json.loads(body)) if verb == "submit" else (cid, user)
                    ok = getattr(state, verb)(*args)
                    return self._send(200, {"ok": True}) if ok else self._send(409, {"detail": "lease not held by this user"})
            return self._send(404, {"detail": "not found"})

    return Handler


def serve(grid: dict[str, Any], port: int = 0, lease_s: float = LEASE_S, *, host: str = "127.0.0.1", max_leases: int | None = None,
          behind_jobs_proxy: bool = False) -> tuple[ThreadingHTTPServer, State]:
    state = State(grid, lease_s, max_leases, behind_jobs_proxy)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--grid", help="default: the repo's config/grid.json")
    ap.add_argument("--harnesses", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--state-out", help="write /api/state here on Ctrl-C")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside an HF Job")
    ap.add_argument("--max-leases", type=int, help="live leases per user (default: as the real coordinator)")
    ap.add_argument("--behind-jobs-proxy", action="store_true", help="served from an HF Job's exposed port: the proxy checks and strips the token")
    a = ap.parse_args(argv)
    grid = json.loads(Path(a.grid or Path(__file__).resolve().parents[2] / "config" / "grid.json").read_text())
    grid["harnesses"] = a.harnesses or grid["harnesses"]
    grid["tasks"] = a.tasks or grid["tasks"]
    grid["trials_per_cell"] = a.trials or grid["trials_per_cell"]
    server, state = serve(grid, a.port, host=a.host, max_leases=a.max_leases, behind_jobs_proxy=a.behind_jobs_proxy)
    print(f"fake coordinator on http://127.0.0.1:{server.server_address[1]} ({len(state.queue)} cells)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        if a.state_out:
            Path(a.state_out).write_text(json.dumps(state.state(), indent=2))


if __name__ == "__main__":
    main()
