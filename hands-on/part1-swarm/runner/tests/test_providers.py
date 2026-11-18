"""Which GLM-5.3-Flash provider serves an attendee: a weighted rendezvous hash of the HF username over the
allowlist in config/providers.json, so the room spreads over providers and each attendee always gets the same one."""

import json
from collections import Counter
from pathlib import Path

import pytest

from aidev_swarm import providers as P

CONFIG = Path(__file__).parents[2] / "config"
ALLOW = [{"name": "novita", "weight": 3}, {"name": "zai-org", "weight": 1}]


def users(n):
    return [f"attendee-{i}" for i in range(n)]


def test_order_is_deterministic_and_a_permutation_of_the_allowlist():
    a = P.order("alice", ALLOW)
    assert a == P.order("alice", ALLOW) == P.order("Alice", ALLOW)  # HF names are case-insensitive
    assert sorted(a) == ["novita", "zai-org"]


def test_first_choices_follow_the_weights():
    first = Counter(P.order(u, ALLOW)[0] for u in users(20_000))
    assert first["novita"] / 20_000 == pytest.approx(0.75, abs=0.015)


def test_three_providers_split_by_weight_too():
    allow = [{"name": "a", "weight": 2}, {"name": "b", "weight": 1}, {"name": "c", "weight": 1}]
    first = Counter(P.order(u, allow)[0] for u in users(20_000))
    assert first["a"] / 20_000 == pytest.approx(0.5, abs=0.015) and first["c"] / 20_000 == pytest.approx(0.25, abs=0.015)


def test_dropping_a_provider_moves_only_its_own_attendees():
    allow3 = [{"name": "a", "weight": 1}, {"name": "b", "weight": 1}, {"name": "c", "weight": 1}]
    allow2 = [p for p in allow3 if p["name"] != "c"]
    for u in users(2_000):
        before = P.order(u, allow3)
        after = P.order(u, allow2)
        assert after == [p for p in before if p != "c"]  # the rest keep their order: the fallback is stable too


def test_zero_weight_providers_are_never_picked():
    allow = [{"name": "a", "weight": 1}, {"name": "off", "weight": 0}]
    assert all(P.order(u, allow) == ["a"] for u in users(200))


def test_cross_implementation_vectors_match_protocol_md():
    # Part 2's client (aidev-evolve) implements the same function: these vectors are in PROTOCOL.md
    for user, want in P.TEST_VECTORS.items():
        assert P.order(user, ALLOW) == want
    protocol = (CONFIG.parent / "PROTOCOL.md").read_text()
    for user, want in P.TEST_VECTORS.items():
        assert f"`{user}` -> {', '.join(want)}" in protocol


def test_the_config_file_lists_weighted_providers_of_the_grid_model():
    cfg = json.loads((CONFIG / "providers.json").read_text())
    grid = json.loads((CONFIG / "grid.json").read_text())
    assert cfg["model"] == grid["model"]
    assert {p["name"] for p in cfg["providers"]} and all(p["weight"] > 0 for p in cfg["providers"])
    assert "provider" not in grid  # one place for the provider choice: providers.json


def test_every_allowlisted_provider_is_admitted_by_the_pinned_leaderboard_pool():
    # the Harness Leaderboard's equivalence policy (models.json `equivalence`): only admitted providers may serve
    from hlb import config as hlb_config

    cfg = json.loads((CONFIG / "providers.json").read_text())
    assert {p["name"] for p in cfg["providers"]} <= set(hlb_config.pool_providers(cfg["model"]))


def meter(*records):
    return "\n".join(json.dumps(r) for r in [{"event": "start"}, *records]) + "\n"


def test_provider_errors_count_rate_limits_server_errors_and_fallbacks():
    text = meter({"path": "/v1/chat/completions", "status": 200}, {"path": "/v1/chat/completions", "status": 429},
                 {"path": "/v1/chat/completions", "status": 503}, {"path": "/v1/chat/completions", "status": 200, "fallback": 429},
                 {"path": "/v1/chat/completions", "status": 400})
    assert P.provider_errors(text) == 3


def test_provider_errors_of_a_broken_log_are_zero():
    assert P.provider_errors("not json\n") == 0
