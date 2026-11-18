"""The /evolve stand-in (devtools/fake_board.py): the aggregation the coordinator's /api/evolve/state will serve."""

import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

import httpx

from aidev_evolve import board

spec = importlib.util.spec_from_file_location("fake_board", Path(__file__).parents[1] / "devtools" / "fake_board.py")
fake_board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake_board)


def ev(user, run, kind, it=None, **kw):
    return {"user": user, "event": {"schema": "aidev-evolve/1", "run_id": run, "kind": kind, **({"iteration": it} if it else {}), **kw}}


def test_state_has_one_curve_per_run_and_a_feed_of_patches_newest_first():
    store = fake_board.Store()
    store.add(**ev("alice", "r1", "seed", val_score=0.25, spend_usd=0.1))
    store.add(**ev("alice", "r1", "iteration", 1, title="Keep output tail", accepted=True, best_val=0.5, val_child=0.5, val_parent=0.25, spend_usd=0.2))
    store.add(**ev("alice", "r1", "iteration", 1, title="Keep output tail", accepted=True, best_val=0.5, val_child=0.5, val_parent=0.25, spend_usd=0.2))  # retry
    store.add(**ev("bob", "r2", "seed", val_score=0.375, spend_usd=0.1))
    store.add(**ev("bob", "r2", "iteration", 1, title="More steps", accepted=False, best_val=0.375, val_child=0.25, val_parent=0.375, spend_usd=0.2))
    st = store.state()
    curves = {r["run_id"]: r["curve"] for r in st["runs"]}
    assert curves == {"r1": [0.25, 0.5], "r2": [0.375, 0.375]}
    assert [f["title"] for f in st["feed"]] == ["More steps", "Keep output tail"]
    assert st["totals"] == {"runs": 2, "iterations": 2, "kept": 1, "spend_usd": 0.4}


def test_the_page_and_the_api_round_trip_over_http():
    srv = fake_board.serve(port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = board.Board(f"http://127.0.0.1:{port}", "hf_test")
        b.submit({"run_id": "r1", "kind": "seed", "val_score": 0.25, "spend_usd": 0.05})
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/evolve/state").read())
        assert st["runs"][0]["curve"] == [0.25]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/evolve").read().decode()
        assert "<svg" in page or "race" in page
    finally:
        srv.shutdown()
