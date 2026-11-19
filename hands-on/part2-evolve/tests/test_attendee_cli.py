"""The attendee's three commands: `check`, `run` (posts to the workshop board, queues events while offline), `replay`."""

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from aidev_evolve import board, cli, loop, proposer, strategy
from aidev_evolve.proposer import ProposerResult

from test_loop import SPLITS, FakeEval, FakeProposer, add_fix


# -- check -----------------------------------------------------------------------------------------

def _probe_run(stdout: str, returncode: int = 0):
    calls = []

    def run(argv, cwd, env, timeout):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    return run, calls


def test_probe_claude_ok_when_it_answers_ok():
    run, calls = _probe_run(json.dumps({"result": "ok", "is_error": False}))
    ok, msg = proposer.probe("claude", run=run, which=lambda n: f"/bin/{n}")
    assert ok and "claude" in msg
    assert calls[0][:2] == ["/bin/claude", "-p"]


def test_probe_reports_a_lost_login_with_the_fix():
    run, _ = _probe_run(json.dumps({"result": "Invalid API key · Please run /login", "is_error": True}), returncode=1)
    ok, msg = proposer.probe("claude", run=run, which=lambda n: n)
    assert not ok and "/login" in msg


def test_probe_reports_a_missing_cli():
    ok, msg = proposer.probe("codex", which=lambda n: None)
    assert not ok and "not found" in msg


def test_probe_codex_ok():
    run, calls = _probe_run("ok\n")
    ok, _ = proposer.probe("codex", run=run, which=lambda n: n)
    assert ok and calls[0][:2] == ["codex", "exec"]


CLAUDE_WARNING = "[mcp-sdk] SEP-2352: stored OAuth credential has no 'issuer' stamp; re-run /login to refresh it\n"
CODEX_STDOUT = ("OpenAI Codex v0.98.0 (research preview)\n--------\nworkdir: /tmp/x\nmodel: gpt-6.2-sol\nprovider: openai\n"
                "approval: never\nsandbox: read-only\nsession id: 019a-4f2e\n--------\nuser\nReply with the single word: ok\n\n"
                "codex\nok\ntokens used\n1,214\n")


def _run_with(stdout: str, stderr: str = "", returncode: int = 0):
    def run(argv, cwd, env, timeout):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return run


def test_probe_tolerates_claudes_warning_line_before_the_json():
    for out, err in [(CLAUDE_WARNING + json.dumps({"result": "ok", "is_error": False}) + "\n", ""),
                     (json.dumps({"result": "ok", "is_error": False}), CLAUDE_WARNING)]:
        ok, msg = proposer.probe("claude", run=_run_with(out, err), which=lambda n: n)
        assert ok, msg


def test_probe_tolerates_codex_header_and_token_footer():
    ok, msg = proposer.probe("codex", run=_run_with(CODEX_STDOUT), which=lambda n: n)
    assert ok, msg
    ok, _ = proposer.probe("codex", run=_run_with(CODEX_STDOUT.replace("codex\nok\n", "codex\nI cannot do that\n")), which=lambda n: n)
    assert not ok  # the header's prompt echo ("... word: ok") is not an answer


def test_the_proposer_run_tolerates_claudes_warning_line(tmp_path):
    out = CLAUDE_WARNING + json.dumps({"result": "done", "is_error": False, "total_cost_usd": 0.2})
    res = proposer.propose("claude", tmp_path, "P", run=lambda argv, cwd, env, timeout: subprocess.CompletedProcess(argv, 0, out, CLAUDE_WARNING),
                           which=lambda n: n)
    assert res.ok and res.cost_usd == pytest.approx(0.2)


def _fake_check_env(monkeypatch, *, proposers=("claude",), board_status=200):
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    monkeypatch.setattr(proposer, "probe", lambda b, **kw: (b in proposers, f"{b}: {'ok' if b in proposers else 'Not logged in'}"))

    def handler(request):
        return httpx.Response(board_status, json={"runs": []})

    monkeypatch.setattr(cli, "_http", lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def test_check_passes_with_hf_login_one_proposer_and_the_board(monkeypatch, capsys):
    _fake_check_env(monkeypatch)
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "alice" in out and "ready" in out.lower()
    assert "aidev-evolve run --proposer claude" in out  # tells the attendee the exact next command


def test_check_fails_without_any_logged_in_proposer(monkeypatch, capsys):
    _fake_check_env(monkeypatch, proposers=())
    assert cli.main(["check"]) == 1
    assert "--proposer manual" in capsys.readouterr().out


def test_check_with_an_explicit_proposer_probes_only_that_one(monkeypatch, capsys):
    _fake_check_env(monkeypatch, proposers=("claude",))
    assert cli.main(["check", "--proposer", "codex"]) == 1


def test_check_does_not_fail_when_only_the_board_is_down(monkeypatch, capsys):
    """The board is a display; the loop runs without it, so a down board is a warning, not a failure."""
    _fake_check_env(monkeypatch, board_status=503)
    assert cli.main(["check"]) == 0
    assert "board" in capsys.readouterr().out.lower()


def test_check_job_runs_one_seed_trial_in_an_hf_job(monkeypatch, capsys):
    _fake_check_env(monkeypatch)
    from aidev_evolve import launch

    seen = {}

    def fake_run_eval(api, token, user, spec, on_record, **kw):
        seen["spec"] = spec
        return launch.EvalResult([{"candidate": "seed", "task": "fix-git", "outcome": "pass", "cost_usd": 0.001}], "job1", None, 50.0, 0.0002)

    monkeypatch.setattr(launch, "run_eval", fake_run_eval)
    assert cli.main(["check", "--job"]) == 0
    assert [(r["tasks"], r["trials"]) for r in seen["spec"]["runs"]] == [(["fix-git"], 1)]
    from aidev_evolve import providers

    user = cli._session()[2]
    assert seen["spec"]["providers"] == providers.for_user(user)  # the check tests the provider the run will use
    out = capsys.readouterr().out
    assert "HF Job" in out and f"GLM provider: {seen['spec']['providers'][0]}" in out


def test_check_without_hf_login_fails_with_the_fix(monkeypatch, capsys):
    def no_session():
        raise SystemExit("No Hugging Face token: run `uvx hf auth login` (or set HF_TOKEN).")

    _fake_check_env(monkeypatch)
    monkeypatch.setattr(cli, "_session", no_session)
    assert cli.main(["check"]) == 1
    assert "hf auth login" in capsys.readouterr().out


# -- run: the workshop board ---------------------------------------------------------------------------

def test_run_posts_to_the_room_board_by_default_and_no_board_turns_it_off():
    p = cli.build_parser()
    assert p.parse_args(["run"]).board == board.ROOM_URL
    assert p.parse_args(["run", "--no-board"]).board is None


def test_run_refuses_to_start_when_the_proposer_does_not_answer(tmp_path, monkeypatch, capsys):
    """Seed calibration costs credits: a logged-out coding agent must stop the run before it, not after."""
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    monkeypatch.setattr(proposer, "probe", lambda b, **kw: (False, f"{b}: Not logged in (run `claude` and type /login)"))
    monkeypatch.setattr(cli, "make_evolver", lambda *a, **k: pytest.fail("must not start the loop"))
    assert cli.main(["run", "--run-dir", str(tmp_path)]) == 1
    assert "/login" in capsys.readouterr().out


def test_run_prints_the_board_url_first(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    monkeypatch.setattr(proposer, "probe", lambda b, **kw: (True, "ok"))
    seen = {}

    class Evo:
        run_dir = tmp_path

        def run(self, iterations=None, heldout_trials=0):
            seen["out"] = capsys.readouterr().out
            return {}

    monkeypatch.setattr(cli, "make_evolver", lambda args, run_dir, write=None: Evo())
    assert cli.main(["run", "--run-dir", str(tmp_path)]) == 0
    assert f"{board.ROOM_URL}/evolve" in seen["out"]


class FlakyTransport:
    """The board: down for the first `down` requests, then up; records what arrived."""

    def __init__(self, down: int):
        self.down, self.arrived, self.calls = down, [], 0

    def __call__(self, request):
        self.calls += 1
        if self.calls <= self.down:
            raise httpx.ConnectError("Wi-Fi dropped")
        self.arrived.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})


def test_queued_board_keeps_events_while_offline_and_delivers_them_in_order_once(tmp_path):
    t = FlakyTransport(down=2)
    q = board.QueuedBoard(board.Board("https://room", "hf_x", client=httpx.Client(transport=httpx.MockTransport(t))), tmp_path / "outbox.jsonl")
    assert q.submit({"kind": "seed", "spend_usd": 0.1}) == 1  # offline: queued
    assert q.submit({"kind": "iteration", "iteration": 1, "spend_usd": 0.2}) == 2  # still offline
    assert q.submit({"kind": "iteration", "iteration": 2, "spend_usd": 0.3}) == 0  # back: everything flushed
    assert [(e["kind"], e.get("iteration")) for e in t.arrived] == [("seed", None), ("iteration", 1), ("iteration", 2)]


def test_queued_board_survives_a_restart(tmp_path):
    """Ctrl-C while offline, then `run --resume`: the queued events are still delivered."""
    down = FlakyTransport(down=99)
    q = board.QueuedBoard(board.Board("https://room", "hf_x", client=httpx.Client(transport=httpx.MockTransport(down))), tmp_path / "outbox.jsonl")
    q.submit({"kind": "seed", "spend_usd": 0.1})
    up = FlakyTransport(down=0)
    q2 = board.QueuedBoard(board.Board("https://room", "hf_x", client=httpx.Client(transport=httpx.MockTransport(up))), tmp_path / "outbox.jsonl")
    assert q2.flush() == 0
    assert [e["kind"] for e in up.arrived] == ["seed"]


def test_queued_board_does_not_retry_an_event_the_board_refused(tmp_path):
    """A 4xx (bad body, bad token) will never succeed: drop it instead of blocking the queue forever."""
    def handler(request):
        return httpx.Response(422, json={"detail": "bad"})

    q = board.QueuedBoard(board.Board("https://room", "hf_x", client=httpx.Client(transport=httpx.MockTransport(handler))), tmp_path / "outbox.jsonl")
    assert q.submit({"kind": "seed"}) == 0


def test_queued_board_final_flush_retries_with_backoff(tmp_path):
    t = FlakyTransport(down=3)
    sleeps = []
    q = board.QueuedBoard(board.Board("https://room", "hf_x", client=httpx.Client(transport=httpx.MockTransport(t))), tmp_path / "outbox.jsonl")
    q.submit({"kind": "heldout"})  # down (1)
    assert q.flush(retries=3, sleep=sleeps.append) == 0  # down (2), down (3), up
    assert len(sleeps) == 2 and [e["kind"] for e in t.arrived] == ["heldout"]


def test_the_loop_says_how_many_events_wait_when_the_board_is_offline(tmp_path):
    t = FlakyTransport(down=99)
    q = board.QueuedBoard(board.Board("https://room", "hf_x", client=httpx.Client(transport=httpx.MockTransport(t))), tmp_path / "outbox.jsonl")
    out = []
    evo = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=FakeEval(), propose=FakeProposer([("Fix it", add_fix)]), board=q,
                       write=out.append, budget_s=150, strategy=strategy.with_knobs(strategy.load("baseline"), val_trials=1, z=1.0))
    evo.run(iterations=1)
    text = "".join(out)
    assert "2 updates queued" in text and evo.state["incumbent"] == "c001"  # the loop goes on


# -- run: what the terminal shows per iteration ----------------------------------------------------

def test_hypothesis_and_coloured_diff_show_before_the_evaluation_starts(tmp_path):
    long = "Hypothesis: the agent stops after ten steps in sparql and financial traces while time is left " + "x" * 300

    class Prop(FakeProposer):
        def __call__(self, work, prompt):
            res = super().__call__(work, prompt)
            (Path(work) / "PATCH.md").write_text(f"# Use the clock\n\n{long}\nChange: y\nExpected: z\n")
            return res

    out = []
    evo = loop.Evolver(tmp_path / "run", splits=SPLITS, evaluate=FakeEval(), propose=Prop([("Use the clock", add_fix)]),
                       write=out.append, budget_s=150, strategy=strategy.with_knobs(strategy.load("baseline"), val_trials=1, z=1.0), color=True)
    evo.run(iterations=1)
    text = "".join(out)
    it = text.index("--- iteration 1")
    hyp, diff, ev = text.index("Hypothesis: the agent stops", it), text.index("\033[32m+# FIX", it), text.index("evaluating", it)
    assert hyp < diff < ev  # read while the Job runs
    hyp_line = text[hyp:text.index("\n", hyp)]
    assert len(hyp_line) <= 160  # one line on a laptop terminal
    assert text.count("+# FIX") == 1  # the result card does not print the diff a second time


def test_a_failed_evaluation_job_ends_run_with_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    from aidev_evolve.launch import EvalJobError

    class Evo:
        run_dir = tmp_path

        def run(self, iterations=None, heldout_trials=0):
            raise EvalJobError("evaluation Job j1 ended without its results (0 received). Last log lines:\ncurl: (22) 404")

    monkeypatch.setattr(cli, "make_evolver", lambda args, run_dir, write=None: Evo())
    assert cli.main(["run", "--run-dir", str(tmp_path), "--proposer", "manual", "--no-board"]) == 1
    out = capsys.readouterr().out
    assert "curl: (22) 404" in out and f"--resume --run-dir {tmp_path}" in out


def test_board_url_comes_from_AIDEV_BOARD_shared_with_the_part1_runner(monkeypatch):
    # one setting for both hands-ons: a dry run points both at the staging Space
    monkeypatch.setenv("AIDEV_BOARD", "https://staging.example")
    p = cli.build_parser()
    assert p.parse_args(["run"]).board == "https://staging.example"
    assert p.parse_args(["check"]).board == "https://staging.example"
    assert p.parse_args(["run", "--board", "https://flag.example"]).board == "https://flag.example"
    assert p.parse_args(["run", "--no-board"]).board is None


def test_without_AIDEV_BOARD_the_workshop_space_is_the_board(monkeypatch):
    monkeypatch.delenv("AIDEV_BOARD", raising=False)
    assert cli.build_parser().parse_args(["check"]).board == board.ROOM_URL
