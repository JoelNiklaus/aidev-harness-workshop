"""`coordinator`: run the swarm coordinator (live) or the offline replay.

    uv run coordinator                         # live; persists to $SWARM_BUCKET when set (needs HF_TOKEN)
    uv run coordinator --replay                # offline fallback: replays recorded results, no network
    uv run coordinator --replay --replay-minutes 10 --port 8000

The same process serves Part 2's evolve board (/evolve); --replay also replays the recorded evolve runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import evolve, paths
from .app import create_app
from .auth import FakeAuth, NoAuth, WhoamiAuth
from .evolve_replay import EvolveReplayer, load_recorded, reference_curves
from .grid import Grid
from .replay import Replayer, load_results
from .sink import BucketSink, NullSink
from .store import LEASE_S, Store

log = logging.getLogger("swarm")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="coordinator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--grid", default=str(paths.grid_path()))
    ap.add_argument("--prices", default=str(paths.prices_path()))
    ap.add_argument("--reference", default=str(paths.reference_path()), help="reference layer (Harness Leaderboard trials)")
    ap.add_argument("--db", default=":memory:", help="SQLite file (rebuilt from the bucket on start)")
    ap.add_argument("--replay", action="store_true", help="offline: stream recorded results into the board, take no claims")
    ap.add_argument("--replay-file", default=None, help="results to replay (default: the reference file)")
    ap.add_argument("--replay-minutes", type=float, default=20.0, help="replay length (swarm results and evolve runs alike)")
    ap.add_argument("--evolve-recorded", default=str(paths.evolve_replay_path()), help="recorded evolve runs (replay and live backdrop)")
    ap.add_argument("--replay-concurrency", type=int, default=150, help="virtual runners in the replay")
    ap.add_argument("--fake-auth", action="store_true", help="load tests only: the bearer token is the username")
    ap.add_argument("--lease-s", type=float, default=LEASE_S, help="lease length; load tests shorten it")
    return ap.parse_args(argv)


@dataclass
class Components:
    app: Any
    store: Store
    auth: Any
    sink: Any
    replayer: Replayer | None
    mode: str
    evolve_store: evolve.EvolveStore
    evolve_replayer: EvolveReplayer | None


def build(args: argparse.Namespace, *, env: dict[str, str] | None = None, hub_api: Any = None) -> Components:
    env = dict(os.environ) if env is None else env
    grid = Grid.load(args.grid)
    prices = json.loads(Path(args.prices).read_text())
    reference = load_results(args.reference) if Path(args.reference).exists() else []
    meta = json.loads(paths.harness_meta_path().read_text())

    if args.fake_auth and env.get("SPACE_ID"):
        raise SystemExit("--fake-auth is for local load tests only; refusing to run it on a Space")
    if args.replay:
        mode, auth, sink = "replay", NoAuth(), NullSink()
    else:
        mode = "live"
        auth = FakeAuth() if args.fake_auth else WhoamiAuth()
        bucket = env.get("SWARM_BUCKET")
        if bucket:
            if hub_api is None:
                from huggingface_hub import HfApi
                hub_api = HfApi(token=env.get("HF_TOKEN"))
            sink = BucketSink(hub_api, bucket)
        else:
            sink = NullSink()
            log.warning("SWARM_BUCKET is not set: results live only in this process")

    store = Store(grid, prices, sink=sink, db_path=args.db, lease_s=args.lease_s)
    n = store.restore()
    if n:
        log.info("restored %d results from %s", n, env.get("SWARM_BUCKET"))
    evolve_store = evolve.EvolveStore(sink=sink)
    n = evolve_store.restore()
    if n:
        log.info("restored %d evolve events", n)
    recorded = load_recorded(args.evolve_recorded) if Path(args.evolve_recorded).exists() else []
    replayer = evolve_replayer = None
    if args.replay:
        replayer = Replayer(store, load_results(args.replay_file or args.reference), minutes=args.replay_minutes, concurrency=args.replay_concurrency)
        evolve_replayer = EvolveReplayer(evolve_store, recorded, minutes=args.replay_minutes)

    @asynccontextmanager
    async def lifespan(_app):
        sink.start()
        for rp in (replayer, evolve_replayer):
            if rp is not None:
                rp.start = rp.clock()  # the replay starts when the server does
                rp.start_thread()
        yield
        for rp in (replayer, evolve_replayer):
            if rp is not None:
                rp.stop()
        sink.close()

    app = create_app(store, auth, reference=reference, meta=meta, mode=mode, lifespan=lifespan, evolve_store=evolve_store,
                     evolve_reference=reference_curves(recorded))
    return Components(app, store, auth, sink, replayer, mode, evolve_store, evolve_replayer)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse(argv)
    c = build(args)
    import uvicorn
    where = f"http://{args.host}:{args.port}/"
    print(f"{c.mode} coordinator on {where} ({len(c.store.grid.cells)} cells)", file=sys.stderr)
    uvicorn.run(c.app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
