import httpx
import pytest

from aidev_evolve import cli


@pytest.fixture(autouse=True)
def _no_board_setting_from_the_shell(monkeypatch):
    """A developer's `export AIDEV_BOARD=<staging>` must not change what the tests see as the default board."""
    monkeypatch.delenv("AIDEV_BOARD", raising=False)


@pytest.fixture(autouse=True)
def _no_network_from_the_cli(monkeypatch):
    """The CLI's own HTTP client (board state, version question) answers 404 unless a test gives it a board."""
    monkeypatch.setattr(cli, "_http", lambda: httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))))
