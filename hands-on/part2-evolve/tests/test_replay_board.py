"""Offline fallback and the room board client."""

import json

import httpx

from aidev_evolve import board, replay


def test_replay_prints_the_recorded_terminal_in_order_with_scaled_capped_pauses(tmp_path):
    (tmp_path / "transcript.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"t": 0.0, "text": "start\n"}, {"t": 10.0, "text": "trial\n"}, {"t": 200.0, "text": "kept\n"}]) + "\n")
    out, sleeps = [], []
    replay.replay(tmp_path, speed=10, max_pause_s=5, write=out.append, sleep=sleeps.append)
    assert out == ["start\n", "trial\n", "kept\n"]
    assert sleeps == [1.0, 5]


def test_bundled_recorded_run_exists_and_replays():
    run = replay.bundled()
    assert (run / "transcript.jsonl").exists()
    out = []
    replay.replay(run, speed=1e9, write=out.append, sleep=lambda s: None)
    assert any("Iteration" in o for o in out)


def test_board_client_posts_events_with_the_token():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    b = board.Board("https://room.example", "hf_x", client=httpx.Client(transport=httpx.MockTransport(handler)))
    b.submit({"kind": "iteration", "iteration": 1, "spend_usd": 0.1})
    assert seen["url"] == "https://room.example/api/evolve/submit" and seen["auth"] == "Bearer hf_x"
    assert seen["body"]["kind"] == "iteration" and seen["body"]["schema"] == board.SCHEMA


def test_the_room_board_is_the_part1_coordinator():
    """One board URL: the --board help names the Space the Part 1 runner defaults to."""
    import re
    from pathlib import Path

    from aidev_evolve import cli

    runner_cli = (Path(__file__).parents[2] / "part1-swarm" / "runner" / "src" / "aidev_swarm" / "cli.py").read_text()
    assert board.ROOM_URL == re.search(r'DEFAULT_COORDINATOR = "([^"]+)"', runner_cli).group(1)
    run = cli.build_parser()._subparsers._group_actions[0].choices["run"]
    assert board.ROOM_URL in next(a.help for a in run._actions if a.dest == "board")
