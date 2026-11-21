"""A client older than the board accepts says so at once (deck request, 21 Nov): the board answers such a client's
events with HTTP 426, and without this line an attendee with an old command never learns why their run is missing."""

import httpx
import pytest

from aidev_evolve import __version__, board, cli, proposer


def client(payload=None, status=200, error=None):
    def handler(request):
        assert request.url.path == "/api/versions"
        if error:
            raise error
        return httpx.Response(status, json=payload if payload is not None else {})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_no_problem_when_the_board_accepts_this_version():
    assert board.version_problem("https://room", http=client({"aidev-evolve": __version__})) is None
    assert board.version_problem("https://room", http=client({"aidev-evolve": "0.1.0"})) is None


def test_a_newer_minimum_is_reported_with_both_versions_and_the_start_page():
    msg = board.version_problem("https://room/", http=client({"aidev-evolve": "9.0.0"}))
    assert msg and "9.0.0" in msg and __version__ in msg and "https://room/evolve/join" in msg


def test_versions_compare_as_numbers_not_as_text():
    assert board.older("0.5.2", "0.5.10") and not board.older("0.10.0", "0.9.9") and not board.older("0.5.2", "0.5.2")


@pytest.mark.parametrize("http", [client(status=404), client(status=503), client({"other": "1.0.0"}), client({"aidev-evolve": "soon"}),
                                  client(error=httpx.ConnectError("no Wi-Fi"))])
def test_an_unreachable_or_older_board_is_never_a_problem(http):
    """The board is a display: a board without the endpoint, a down board or no network must not stop or scare anyone."""
    assert board.version_problem("https://room", http=http) is None


def _fake_run(monkeypatch, tmp_path, problem):
    monkeypatch.setattr(cli, "_session", lambda: (object(), "hf_dummy", "alice"))
    monkeypatch.setattr(proposer, "probe", lambda b, **kw: (True, "ok"))
    monkeypatch.setattr(board, "version_problem", lambda url, **kw: problem)

    class Evo:
        run_dir = tmp_path

        def run(self, iterations=None, heldout_trials=0):
            return {}

    monkeypatch.setattr(cli, "make_evolver", lambda args, run_dir, write=None: Evo())


def test_run_says_at_the_start_that_the_board_will_not_show_this_run(monkeypatch, tmp_path, capsys):
    _fake_run(monkeypatch, tmp_path, "this aidev-evolve (0.5.2) is older than the workshop board accepts (0.6.0)")
    assert cli.main(["run", "--run-dir", str(tmp_path)]) == 0  # the loop still runs: the board is a display
    out = capsys.readouterr().out
    assert "older than the workshop board accepts" in out and out.index("older than") < out.index("Run folder")


def test_run_says_nothing_when_the_version_is_fine_or_the_board_is_off(monkeypatch, tmp_path, capsys):
    _fake_run(monkeypatch, tmp_path, None)
    cli.main(["run", "--run-dir", str(tmp_path)])
    assert "older than" not in capsys.readouterr().out
    monkeypatch.setattr(board, "version_problem", lambda url, **kw: pytest.fail("no board, no question"))
    assert cli.main(["run", "--run-dir", str(tmp_path), "--no-board"]) == 0


def test_check_fails_with_the_fix_when_this_version_is_too_old(monkeypatch, capsys):
    from test_attendee_cli import _fake_check_env

    _fake_check_env(monkeypatch)
    monkeypatch.setattr(board, "version_problem", lambda url, **kw: "this aidev-evolve (0.5.2) is older than the workshop board accepts (0.6.0): copy the command again from https://room/evolve/join")
    assert cli.main(["check"]) == 1
    assert "copy the command again" in capsys.readouterr().out
