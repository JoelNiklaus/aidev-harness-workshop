"""The coordinator client against an in-memory fake of the PROTOCOL.md HTTP API (httpx.MockTransport)."""

import json

import httpx
import pytest

from aidev_swarm.coordinator import Coordinator, CoordinatorError, LeaseLost

TOKEN = "hf_dummy_token_for_tests"


class FakeServer:
    """Answers only the routes the protocol defines; anything else is a 404 so an unplanned call fails the test."""

    def __init__(self, cells=None):
        self.cells = list(cells if cells is not None else [{"cell_id": "pi|t1|0", "harness": "pi", "task": "t1", "trial": 0, "model": "m",
                                                             "lease_expires_at": "2026-11-17T12:00:00Z", "run_config": {"provider": "novita"}}])
        self.calls = []
        self.submitted = {}
        self.foreign = set()  # cell ids whose lease another user holds

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path, dict(request.url.params)))
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        path = request.url.path
        if request.method == "POST" and path == "/api/claim":
            if not self.cells:
                return httpx.Response(204)
            out, self.cells = self.cells[0], self.cells[1:]
            return httpx.Response(200, json=out)
        for verb in ("heartbeat", "submit", "release"):
            prefix = f"/api/{verb}/"
            if request.method == "POST" and path.startswith(prefix):
                cell = path[len(prefix):]
                if cell in self.foreign:
                    return httpx.Response(409, json={"detail": "lease held by another user"})
                if verb == "submit":
                    self.submitted[cell] = json.loads(request.content)
                return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def client(server, **kw):
    return Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(server.handler), **kw)


def test_claim_returns_the_cell():
    s = FakeServer()
    cell = client(s).claim()
    assert cell["cell_id"] == "pi|t1|0" and cell["run_config"] == {"provider": "novita"}
    assert s.calls == [("POST", "/api/claim", {})]


def test_claim_returns_none_when_the_grid_is_full():
    assert client(FakeServer(cells=[])).claim() is None


def test_claim_with_a_limit_sends_it_and_returns_one_cell():
    s = FakeServer(cells=[{"cell_id": "a"}, {"cell_id": "b"}])
    assert client(s).claim(limit=3)["cell_id"] == "a"
    assert s.calls[-1] == ("POST", "/api/claim", {"limit": "3"})


def test_cell_ids_with_pipes_are_url_quoted_and_reach_the_server_intact():
    s = FakeServer()
    client(s).heartbeat("pi|t1|0")
    assert s.calls[-1][1] == "/api/heartbeat/pi|t1|0"


def test_submit_posts_the_result_json():
    s = FakeServer()
    client(s).submit("pi|t1|0", {"cell_id": "pi|t1|0", "passed": True})
    assert s.submitted["pi|t1|0"] == {"cell_id": "pi|t1|0", "passed": True}


def test_heartbeat_on_a_lease_held_by_someone_else_raises_lease_lost():
    s = FakeServer()
    s.foreign.add("pi|t1|0")
    with pytest.raises(LeaseLost):
        client(s).heartbeat("pi|t1|0")


def test_server_error_raises_coordinator_error_without_the_token_in_the_message():
    def boom(request):
        return httpx.Response(500, text="internal")

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(boom), retries=1, backoff_s=0)
    with pytest.raises(CoordinatorError) as e:
        c.claim()
    assert TOKEN not in str(e.value)


def test_transient_failures_are_retried_then_succeed():
    attempts = []

    def flaky(request):
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("wifi")
        return httpx.Response(200, json={"ok": True})

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(flaky), retries=5, backoff_s=0)
    c.submit("x", {"cell_id": "x"})
    assert len(attempts) == 3


class FakeClock:
    """Fake time for the retry loop; raises after many sleeps so a missing give-up can never hang the suite."""

    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        assert len(self.sleeps) < 1000, "retry loop never gave up"
        self.t += s


def rate_limited_for(seconds, clock, retry_after=None):
    """The HF edge's per-IP 429 for `seconds` of (fake) time, then the coordinator answers."""
    def handler(request):
        if clock.t < seconds:
            headers = {"Retry-After": str(retry_after)} if retry_after else {}
            return httpx.Response(429, text="Too Many Requests", headers=headers)
        return httpx.Response(200, json={"ok": True})

    return handler


def test_submit_keeps_retrying_through_a_29_minute_rate_limit():
    clock = FakeClock()
    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(rate_limited_for(29 * 60, clock)), sleep=clock.sleep, clock=clock.time)
    c.submit("x", {"cell_id": "x"})
    assert clock.t >= 29 * 60
    assert max(clock.sleeps) <= 60, "backoff must stay capped so the result lands soon after the block lifts"


def test_submit_gives_up_after_its_budget_of_at_least_30_minutes():
    clock = FakeClock()
    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(rate_limited_for(10**9, clock)), sleep=clock.sleep, clock=clock.time)
    with pytest.raises(CoordinatorError, match="429"):
        c.submit("x", {"cell_id": "x"})
    assert 30 * 60 <= clock.t <= 40 * 60


def test_heartbeat_gives_up_fast_so_the_next_beat_can_try_again():
    clock = FakeClock()
    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(rate_limited_for(10**9, clock)), sleep=clock.sleep, clock=clock.time)
    with pytest.raises(CoordinatorError):
        c.heartbeat("x")
    assert clock.t <= 120


def test_retry_after_header_is_honoured():
    clock = FakeClock()
    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(rate_limited_for(1, clock, retry_after=45)), sleep=clock.sleep, clock=clock.time)
    c.submit("x", {"cell_id": "x"})
    assert clock.sleeps == [45]


def test_claim_beyond_the_lease_limit_raises_at_limit_with_the_held_cells():
    from aidev_swarm.coordinator import AtLimit

    def handler(request):
        return httpx.Response(409, json={"detail": "you already hold 1 active lease(s): pi__t1__0", "held": ["pi__t1__0"]})

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(AtLimit) as e:
        c.claim()
    assert e.value.held == ["pi__t1__0"]


def test_bad_request_422_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(422, json={"detail": "result does not match cell"})

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(CoordinatorError, match="422"):
        c.submit("x", {"cell_id": "x"})
    assert len(calls) == 1


def test_every_request_names_the_runner_version_in_the_user_agent():
    """The coordinator takes claims only from runners that speak the short-lease protocol (PROTOCOL.md)."""
    from aidev_swarm import __version__

    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, json={"ok": True})

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(handler))
    c.heartbeat("x")
    assert seen == [f"aidev-swarm/{__version__}"]


def test_a_204_with_retry_after_means_every_free_cell_is_taken_for_now():
    from aidev_swarm.coordinator import AllTaken

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(lambda r: httpx.Response(204, headers={"Retry-After": "30"})))
    with pytest.raises(AllTaken) as e:
        c.claim()
    assert e.value.retry_after == 30


def test_a_426_old_runner_refusal_is_not_retried_and_shows_the_coordinators_reason():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(426, json={"detail": "this runner is too old (needs aidev-swarm >= 0.4.0)"})

    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(CoordinatorError, match="too old"):
        c.claim()
    assert len(calls) == 1


def test_a_quick_release_on_the_way_out_gives_up_within_seconds():
    """A cancelled Job has seconds before SIGKILL; if the release does not land, the short lease expires instead."""
    clock = FakeClock()
    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(rate_limited_for(10**9, clock)), sleep=clock.sleep, clock=clock.time)
    with pytest.raises(CoordinatorError):
        c.release("x", quick=True)
    assert clock.t <= 8


def test_heartbeat_retries_end_before_the_next_beat_is_due():
    from aidev_swarm.cli import HEARTBEAT_S

    clock = FakeClock()
    c = Coordinator("https://swarm.example", TOKEN, transport=httpx.MockTransport(rate_limited_for(10**9, clock)), sleep=clock.sleep, clock=clock.time)
    with pytest.raises(CoordinatorError):
        c.heartbeat("x")
    assert clock.t <= HEARTBEAT_S
