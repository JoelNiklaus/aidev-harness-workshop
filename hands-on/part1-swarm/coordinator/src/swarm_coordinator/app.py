"""The coordinator's HTTP surface (PROTOCOL.md): claim, heartbeat, submit, release, state, board, join page."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import join, paths
from .auth import AuthUnavailable, Unauthorized
from .state import build_state
from .store import AtLimit, Conflict, Invalid, NotFound, Store

log = logging.getLogger("swarm.app")

STATE_TTL_S = 2.0
BOOT_OPEN = b'<script id="boot" type="application/json">'
BOOT_SLOT = BOOT_OPEN + b"null</script>"


def create_app(store: Store, auth: Any, *, reference: list[dict[str, Any]], meta: dict[str, dict[str, str]], mode: str = "live",
               mono: Callable[[], float] = time.monotonic, state_ttl_s: float = STATE_TTL_S, lifespan: Any = None,
               runner_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="aidev swarm coordinator", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    board = resources.files("swarm_coordinator").joinpath("static/board.html").read_bytes()
    if BOOT_SLOT not in board:
        raise RuntimeError("board.html lacks the boot state slot")
    join_page = join.page(runner_dir or paths.runner_dir())  # rendered once: a README without the wheel URL fails at start
    cache: dict[str, Any] = {"at": None, "body": b"", "etag": ""}

    @app.exception_handler(Unauthorized)
    async def _401(_: Request, e: Unauthorized) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=401)

    @app.exception_handler(AuthUnavailable)
    async def _503(_: Request, e: AuthUnavailable) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=503)

    @app.exception_handler(Conflict)
    async def _409(_: Request, e: Conflict) -> JSONResponse:
        body: dict[str, Any] = {"detail": str(e)}
        if isinstance(e, AtLimit):
            body["held"] = e.held
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
        return who or await run_in_threadpool(auth.user, header)

    @app.post("/api/claim")
    async def claim(request: Request, n: int = 1) -> Response:
        who = await user(request)
        got = store.claim(who, n)
        if not got:
            return Response(status_code=204)
        log.info("claim %s -> %s", who, ",".join(c["cell_id"] for c in got))
        return JSONResponse(got if n > 1 else got[0])

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

    def state_body() -> bytes:
        """The board payload, rebuilt at most every state_ttl_s whatever the number of viewers."""
        now = mono()
        if cache["at"] is None or now - cache["at"] >= state_ttl_s:
            body = json.dumps(build_state(store.grid, store.snapshot(), reference, meta, mode=mode), separators=(",", ":")).encode()
            cache.update(at=now, body=body, etag='"' + hashlib.sha1(body).hexdigest()[:16] + '"')
        return cache["body"]

    @app.get("/api/state")
    async def state(request: Request) -> Response:
        state_body()
        headers = {"ETag": cache["etag"], "Cache-Control": f"public, max-age={int(state_ttl_s)}"}
        if request.headers.get("if-none-match") == cache["etag"]:
            return Response(status_code=304, headers=headers)
        return Response(cache["body"], media_type="application/json", headers=headers)

    @app.get("/")
    async def index() -> HTMLResponse:
        # the current state rides along, so the first full draw needs no second round trip;
        # "</" is escaped ("<\/" is the same JSON string) so no value can close the script tag
        boot = state_body().replace(b"</", b"<\\/")
        return HTMLResponse(board.replace(BOOT_SLOT, BOOT_OPEN + boot + b"</script>", 1), headers={"Cache-Control": "no-cache"})

    @app.get(join.PATH)
    async def join_() -> HTMLResponse:
        return HTMLResponse(join_page, headers={"Cache-Control": "public, max-age=60"})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "mode": mode, "version": store.version}

    return app
