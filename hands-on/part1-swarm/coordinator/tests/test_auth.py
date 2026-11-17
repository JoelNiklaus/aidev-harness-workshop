from __future__ import annotations

import logging

import pytest

from swarm_coordinator.auth import AuthUnavailable, Unauthorized, WhoamiAuth

TOKEN = "hf_SECRETsecretSECRETsecret1234"


class FakeWhoami:
    def __init__(self, users=None, down=False):
        self.users = users or {TOKEN: "alice"}
        self.calls = 0
        self.down = down

    def __call__(self, token: str) -> str | None:
        self.calls += 1
        if self.down:
            raise AuthUnavailable("hub unreachable")
        return self.users.get(token)


class Mono:
    t = 0.0

    def __call__(self):
        return self.t


def test_resolves_user_and_caches_per_token():
    f = FakeWhoami()
    a = WhoamiAuth(fetch=f)
    assert a.user(f"Bearer {TOKEN}") == "alice"
    assert a.user(f"Bearer {TOKEN}") == "alice"
    assert f.calls == 1


def test_cache_expires():
    f, clock = FakeWhoami(), Mono()
    a = WhoamiAuth(fetch=f, ttl_s=60, clock=clock)
    a.user(f"Bearer {TOKEN}")
    clock.t = 61
    a.user(f"Bearer {TOKEN}")
    assert f.calls == 2


@pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer ", "bearer"])
def test_missing_or_malformed_header_is_unauthorized(header):
    with pytest.raises(Unauthorized):
        WhoamiAuth(fetch=FakeWhoami()).user(header)


def test_unknown_token_is_unauthorized_and_negative_cached():
    f = FakeWhoami()
    a = WhoamiAuth(fetch=f)
    for _ in range(3):
        with pytest.raises(Unauthorized):
            a.user("Bearer hf_wrong")
    assert f.calls == 1


def test_hub_down_is_unavailable_not_cached():
    f = FakeWhoami(down=True)
    a = WhoamiAuth(fetch=f)
    with pytest.raises(AuthUnavailable):
        a.user(f"Bearer {TOKEN}")
    f.down = False
    assert a.user(f"Bearer {TOKEN}") == "alice"


def test_peek_answers_from_the_cache_only():
    f = FakeWhoami()
    a = WhoamiAuth(fetch=f)
    assert a.peek(f"Bearer {TOKEN}") is None and f.calls == 0
    a.user(f"Bearer {TOKEN}")
    assert a.peek(f"Bearer {TOKEN}") == "alice" and f.calls == 1


def test_token_is_never_kept_or_logged(caplog):
    caplog.set_level(logging.DEBUG)
    a = WhoamiAuth(fetch=FakeWhoami())
    a.user(f"Bearer {TOKEN}")
    with pytest.raises(Unauthorized):
        a.user("Bearer hf_otherSECRETtoken")
    assert TOKEN not in repr(a.__dict__) and "hf_otherSECRETtoken" not in repr(a.__dict__)
    assert "SECRET" not in caplog.text
