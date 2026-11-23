"""The coordinator's HTTP surface (PROTOCOL.md): claim, heartbeat, submit, release, state, board, join page;
Part 2's evolve board: /api/evolve/submit, /api/evolve/state, /api/evolve/diff/..., /evolve, /evolve/join (start page)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from . import evolve, join, paths
from .auth import AuthUnavailable, Forbidden, Unauthorized
from .state import build_state
from .store import AtLimit, Conflict, Invalid, NotFound, Store

log = logging.getLogger("swarm.app")

STATE_TTL_S = 2.0
EVOLVE_PER_MINUTE, EVOLVE_BURST = 6, 10  # an attendee sends ~1 event per 4 min; a resumed run may flush a few at once
CLAIM_AGAIN_S = 30  # every free cell is taken, but a leased one returns if its runner dies: ask again after this
# Runners older than 0.4.0 heartbeat every 2 min against a 90 s lease, so they would lose their cells; 0.4.0 sends every
# trial to zai-org and records no provider (PROTOCOL.md "Providers"). Up to 0.4.2 the pinned Harness Leaderboard code
# (hlb 0.1.7) did not stop an agent whose time was up, so it could keep working while the tests ran and pass on that
# work (fixed in hlb 0.1.9, 2026-11-19; 0.4.3 pins hlb 0.1.11). None of them may claim.
MIN_RUNNER = (0, 4, 3)
RUNNER_UA = re.compile(r"^aidev-swarm/(\d+)\.(\d+)\.(\d+)")
# Part 2: up to 0.5.1 the evaluation pinned the same hlb 0.1.7 (and the client sent no version); 0.5.3 pins hlb 0.1.11
# and names itself on every event; 0.6.0 is the workshop run that fits the session (one Job; a 0.5.3 run takes 30 min in a
# full room and ends after the ranking is read). Both minimums are the frozen workshop versions (a test compares them with the packages).
MIN_EVOLVE = (0, 6, 0)
EVOLVE_UA = re.compile(r"^aidev-evolve/(\d+)\.(\d+)\.(\d+)")
# Staging only (SWARM_LOADTEST_USERS, cli.build): an allowed account may act under load-test names, one per simulated
# attendee, so one account can stand in for a room (each name has its own share, slow start and rate limit).
LOADTEST_HEADER = "x-aidev-loadtest-as"
LOADTEST_NAME = re.compile(r"^lt-[a-z0-9][a-z0-9-]{0,29}$")
BOOT_OPEN = b'<script id="boot" type="application/json">'
BOOT_SLOT = BOOT_OPEN + b"null</script>"


def _at_least(pattern: re.Pattern[str], minimum: tuple[int, int, int], user_agent: str | None) -> bool:
    m = pattern.match(user_agent or "")
    return bool(m) and tuple(int(g) for g in m.groups()) >= minimum


def runner_ok(user_agent: str | None) -> bool:
    """A claim must come from a runner (User-Agent aidev-swarm/X.Y.Z) that speaks the short-lease protocol."""
    return _at_least(RUNNER_UA, MIN_RUNNER, user_agent)


def evolve_ok(user_agent: str | None) -> bool:
    """A Part 2 event must come from the frozen aidev-evolve or newer (User-Agent aidev-evolve/X.Y.Z)."""
    return _at_least(EVOLVE_UA, MIN_EVOLVE, user_agent)


class CachedJson:
    """A JSON payload rebuilt at most every ttl_s whatever the number of viewers, with its ETag."""

    def __init__(self, build: Callable[[], Any], mono: Callable[[], float], ttl_s: float):
        self.build, self.mono, self.ttl_s = build, mono, ttl_s
        self.at: float | None = None
        self.body, self.etag = b"", ""

    def get(self) -> bytes:
        now = self.mono()
        if self.at is None or now - self.at >= self.ttl_s:
            self.body = json.dumps(self.build(), separators=(",", ":")).encode()
            self.at, self.etag = now, '"' + hashlib.sha1(self.body).hexdigest()[:16] + '"'
        return self.body

    def response(self, request: Request) -> Response:
        self.get()
        headers = {"ETag": self.etag, "Cache-Control": f"public, max-age={int(self.ttl_s)}"}
        if request.headers.get("if-none-match") == self.etag:
            return Response(status_code=304, headers=headers)
        return Response(self.body, media_type="application/json", headers=headers)

    def page(self, html: bytes) -> HTMLResponse:
        # the current state rides along, so the first full draw needs no second round trip;
        # "</" is escaped ("<\/" is the same JSON string) so no value can close the script tag
        boot = self.get().replace(b"</", b"<\\/")
        return HTMLResponse(html.replace(BOOT_SLOT, BOOT_OPEN + boot + b"</script>", 1), headers={"Cache-Control": "no-cache"})


def _static(name: str) -> bytes:
    page = resources.files("swarm_coordinator").joinpath(f"static/{name}").read_bytes()
    if BOOT_SLOT not in page:
        raise RuntimeError(f"{name} lacks the boot state slot")
    return page


def create_app(store: Store, auth: Any, *, reference: list[dict[str, Any]], meta: dict[str, dict[str, str]], mode: str = "live",
               mono: Callable[[], float] = time.monotonic, state_ttl_s: float = STATE_TTL_S, lifespan: Any = None,
               runner_dir: Path | None = None, evolve_store: evolve.EvolveStore | None = None,
               evolve_reference: list[dict[str, Any]] | None = None, evolve_limiter: evolve.RateLimiter | None = None,
               evolve_readme: Path | None = None, loadtest_users: frozenset[str] = frozenset(),
               coupons: dict[str, str] | None = None) -> FastAPI:
    app = FastAPI(title="aidev swarm coordinator", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=500)  # the pages fit the first round trip through the HF edge
    board, evolve_page = _static("board.html"), _static("evolve.html")
    coupons = coupons or {}  # each session's shared coupon link, shown on its start page (cli.room_settings)
    join_page = join.page(runner_dir or paths.runner_dir(), coupons.get("part1"))  # rendered once: a README without the wheel URL fails at start
    evolve_join_page = join.evolve_page(evolve_readme or paths.evolve_dir() / "README.md", coupons.get("part2"))
    evolve_store = evolve_store or evolve.EvolveStore()
    evolve_limiter = evolve_limiter or evolve.RateLimiter(per_minute=EVOLVE_PER_MINUTE, burst=EVOLVE_BURST)
    swarm_state = CachedJson(lambda: build_state(store.grid, store.snapshot(), reference, meta, mode=mode), mono, state_ttl_s)
    ranks = evolve.RankCache(clock=mono)
    evolve_state = CachedJson(lambda: evolve.build_state(evolve_store.events(), evolve_reference or [], mode=mode, rank_cache=ranks), mono, state_ttl_s)

    @app.exception_handler(Unauthorized)
    async def _401(_: Request, e: Unauthorized) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=401)

    @app.exception_handler(Forbidden)
    async def _403(_: Request, e: Forbidden) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=403)

    @app.exception_handler(AuthUnavailable)
    async def _503(_: Request, e: AuthUnavailable) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=503)

    @app.exception_handler(Conflict)
    async def _409(_: Request, e: Conflict) -> JSONResponse:
        body: dict[str, Any] = {"detail": str(e)}
        if isinstance(e, AtLimit):
            body["held"], body["share"] = e.held, e.share
        return JSONResponse(body, status_code=409)

    @app.exception_handler(Invalid)
    async def _422(_: Request, e: Invalid) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=422)

    @app.exception_handler(NotFound)
    async def _404(_: Request, e: NotFound) -> JSONResponse:
        return JSONResponse({"detail": f"no cell {e.args[0]!r} in the grid"}, status_code=404)

    async def user(request: Request) -> str:
        # Store operations take well under a millisecond, so every endpoint runs on the event loop;
        # only a whoami call to the Hub (a token not yet cached) goes to a worker thread.
        header = request.headers.get("authorization")
        peek = getattr(auth, "peek", None)
        who = peek(header) if peek else None
        who = who or await run_in_threadpool(auth.user, header)
        name = request.headers.get(LOADTEST_HEADER)
        if name is None:
            return who
        if who.lower() not in loadtest_users:
            raise Forbidden("load-test names are off on this board")
        if not LOADTEST_NAME.match(name):
            raise Invalid("a load-test name looks like lt-<lowercase letters, digits, dashes> (at most 33 characters)")
        return name

    @app.post("/api/claim")
    async def claim(request: Request, limit: int = 1) -> Response:
        who = await user(request)
        if not runner_ok(request.headers.get("user-agent")):
            need = ".".join(map(str, MIN_RUNNER))
            return JSONResponse({"detail": f"this runner is too old for the workshop coordinator (needs aidev-swarm >= {need}); "
                                           "launch again with the current command from the board's /join page"}, status_code=426)
        got = store.claim(who, limit)
        if not got:
            # nothing free: ask again later while leased cells may still come back, stop when the grid is full
            headers = {"Retry-After": str(CLAIM_AGAIN_S)} if store.leased_open() else None
            return Response(status_code=204, headers=headers)
        log.info("claim %s -> %s", who, ",".join(c["cell_id"] for c in got))
        return JSONResponse(got[0])

    @app.post("/api/heartbeat/{cell_id}")
    async def heartbeat(cell_id: str, request: Request) -> dict[str, Any]:
        return store.heartbeat(cell_id, await user(request))

    @app.post("/api/submit/{cell_id}")
    async def submit(cell_id: str, request: Request) -> dict[str, Any]:
        who = await user(request)
        try:
            body = json.loads(await request.body())
        except ValueError:
            raise Invalid("body must be the Result JSON") from None
        stored = store.submit(cell_id, who, body)
        log.info("submit %s %s %s", who, cell_id, "pass" if stored["passed"] else "fail")
        return {"ok": True, "cell_id": cell_id, "passed": stored["passed"], "cost_usd": stored["cost_usd"]}

    @app.post("/api/release/{cell_id}")
    async def release(cell_id: str, request: Request) -> dict[str, Any]:
        store.release(cell_id, await user(request))
        return {"ok": True, "cell_id": cell_id}

    @app.get("/api/state")
    async def state(request: Request) -> Response:
        return swarm_state.response(request)

    @app.get("/")
    async def index() -> HTMLResponse:
        return swarm_state.page(board)

    # ---- Part 2: the evolve board ----------------------------------------------------------------

    @app.post("/api/evolve/submit")
    async def evolve_submit(request: Request) -> Response:
        if int(request.headers.get("content-length") or 0) > evolve.MAX_BODY_BYTES:
            return JSONResponse({"detail": f"event larger than {evolve.MAX_BODY_BYTES} bytes"}, status_code=413)
        who = await user(request)
        if not evolve_ok(request.headers.get("user-agent")):
            need = ".".join(map(str, MIN_EVOLVE))
            return JSONResponse({"detail": f"this aidev-evolve is too old for the workshop board (needs aidev-evolve >= {need}); "
                                           f"start your run with the current command from the board's {join.EVOLVE_PATH} page"}, status_code=426)
        wait = evolve_limiter.allow(who)
        if wait:
            return JSONResponse({"detail": "too many events; retry later"}, status_code=429, headers={"Retry-After": str(math.ceil(wait))})
        raw = await request.body()
        if len(raw) > evolve.MAX_BODY_BYTES:  # a chunked body without Content-Length
            return JSONResponse({"detail": f"event larger than {evolve.MAX_BODY_BYTES} bytes"}, status_code=413)
        try:
            body = json.loads(raw)
        except ValueError:
            raise Invalid("body must be one evolve event as JSON") from None
        stored = evolve_store.add(who, evolve.validate(body))
        log.info("evolve %s %s %s %s", who, stored["run_id"], stored["kind"], stored["iteration"])
        return JSONResponse({"ok": True})

    @app.get("/api/evolve/state")
    async def evolve_state_(request: Request) -> Response:
        return evolve_state.response(request)

    @app.get("/api/evolve/diff/{who}/{run_id}/{iteration}")
    async def evolve_diff(who: str, run_id: str, iteration: int) -> Response:
        diff = evolve_store.diff(who, run_id, iteration)
        if diff is None:
            return JSONResponse({"detail": "no such patch"}, status_code=404)
        return PlainTextResponse(diff, headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/evolve")
    async def evolve_index() -> HTMLResponse:
        return evolve_state.page(evolve_page)

    @app.get(join.PATH)
    async def join_() -> HTMLResponse:
        return HTMLResponse(join_page, headers={"Cache-Control": "public, max-age=60"})

    @app.get(join.EVOLVE_PATH)
    async def evolve_join() -> HTMLResponse:
        return HTMLResponse(evolve_join_page, headers={"Cache-Control": "public, max-age=60"})

    @app.get("/api/versions")
    async def versions() -> dict[str, str]:
        """The oldest clients the board takes (aidev-evolve asks at the start of a run and says so when it is older)."""
        return {"aidev-swarm": ".".join(map(str, MIN_RUNNER)), "aidev-evolve": ".".join(map(str, MIN_EVOLVE))}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "mode": mode, "version": store.version}

    return app
