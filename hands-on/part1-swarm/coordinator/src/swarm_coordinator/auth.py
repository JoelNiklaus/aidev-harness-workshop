"""Bearer-token auth through the Hub's whoami-v2, cached per token.

The cache is keyed by the token's SHA-256, so the process never keeps a token, and nothing here
logs one. Unknown tokens are cached briefly as unknown (a client retry loop costs one Hub call per
minute, not one per request); a Hub outage is never cached.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable

import httpx

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


class Unauthorized(Exception):
    """401: no usable bearer token, or the Hub does not know it."""


class Forbidden(Exception):
    """403: the token is fine, but this user may not do that."""


class AuthUnavailable(Exception):
    """503: the Hub could not be asked."""


def whoami_http(token: str, *, url: str = WHOAMI_URL, timeout_s: float = 10.0) -> str | None:
    """The token's username; None when the Hub rejects the token."""
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout_s)
    except httpx.HTTPError as e:
        raise AuthUnavailable(f"whoami-v2 unreachable ({type(e).__name__})") from None
    if r.status_code in (401, 403):
        return None
    if r.status_code != 200:
        raise AuthUnavailable(f"whoami-v2 answered HTTP {r.status_code}")
    name = r.json().get("name")
    return name or None


class WhoamiAuth:
    def __init__(self, fetch: Callable[[str], str | None] = whoami_http, *, ttl_s: float = 3600.0, neg_ttl_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self._fetch, self._ttl, self._neg_ttl, self._clock = fetch, ttl_s, neg_ttl_s, clock
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(authorization: str | None) -> tuple[str, str]:
        scheme, _, token = (authorization or "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise Unauthorized("send Authorization: Bearer <your HF token>")
        return token, hashlib.sha256(token.encode()).hexdigest()

    def peek(self, authorization: str | None) -> str | None:
        """The cached username, without asking the Hub (None on a miss), so the event loop never waits on it."""
        _, key = self._key(authorization)
        with self._lock:
            hit = self._cache.get(key)
        if hit is None or hit[1] <= self._clock():
            return None
        if hit[0] is None:
            raise Unauthorized("the Hub does not accept this token")
        return hit[0]

    def user(self, authorization: str | None) -> str:
        token, key = self._key(authorization)
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
        if hit is None or hit[1] <= now:
            name = self._fetch(token)  # AuthUnavailable propagates uncached
            hit = (name, now + (self._ttl if name else self._neg_ttl))
            with self._lock:
                self._cache[key] = hit
        if hit[0] is None:
            raise Unauthorized("the Hub does not accept this token")
        return hit[0]


class FakeAuth:
    """Load tests only: the bearer token is the username. Refused on a Space (cli.py)."""

    def peek(self, authorization: str | None) -> str:
        return self.user(authorization)

    def user(self, authorization: str | None) -> str:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise Unauthorized("send Authorization: Bearer <user>")
        return token.strip()


class NoAuth:
    """Replay mode: the board is read-only, every POST is refused."""

    def peek(self, authorization: str | None) -> str:
        return self.user(authorization)

    def user(self, authorization: str | None) -> str:
        raise AuthUnavailable("this coordinator replays recorded results; it takes no claims")
