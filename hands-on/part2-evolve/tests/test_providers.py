"""One GLM-5.3-Flash provider per attendee (the workshop's shared allowlist), with fallback inside an evaluation Job."""

import json
import threading
from pathlib import Path

from aidev_evolve import providers

REPO_CONFIG = Path(__file__).resolve().parents[2] / "part1-swarm" / "config" / "providers.json"


def test_packaged_list_is_a_copy_of_the_workshop_list():
    packaged = json.loads(Path(providers.__file__).with_name("data").joinpath("providers.json").read_text())
    assert packaged == json.loads(REPO_CONFIG.read_text())  # one source: ../part1-swarm/config/providers.json


def test_order_matches_the_part1_runner_check_values():
    plist = providers.load()
    for user, want in providers.TEST_VECTORS.items():
        assert providers.order(user, plist) == want, user
    assert providers.TEST_VECTORS == {"alice": ["novita", "zai-org"], "dave": ["zai-org", "novita"], "JoelNiklaus": ["novita", "zai-org"]}


def test_order_ignores_case_and_leaves_out_weight_zero():
    plist = [{"name": "a", "weight": 1}, {"name": "b", "weight": 0}, {"name": "c", "weight": 2}]
    assert providers.order("Bob", plist) == providers.order("bob", plist)
    assert "b" not in providers.order("bob", plist) and len(providers.order("bob", plist)) == 2


def test_weights_set_each_providers_share_of_attendees():
    plist = providers.load()
    firsts = [providers.order(f"user{i}", plist)[0] for i in range(4000)]
    assert 0.71 < firsts.count("novita") / len(firsts) < 0.79  # weight 3 of 4


def test_pool_falls_back_per_call_and_moves_the_job_after_three_fallbacks():
    pool = providers.Pool(["novita", "zai-org"])
    assert pool.first() == "novita"
    assert pool.fallback("novita") == "zai-org" and pool.first() == "novita"
    pool.fallback("novita")
    assert pool.first() == "novita"
    assert pool.fallback("novita") == "zai-org"
    assert pool.first() == "zai-org"  # the third fallback in this Job: the rest of the Job starts on the next provider
    assert pool.fallback("zai-org") == "novita"


def test_a_single_provider_has_nothing_to_fall_back_to():
    pool = providers.Pool(["zai-org"])
    assert pool.fallback("zai-org") is None and pool.first() == "zai-org"


def test_trials_of_one_job_share_one_pool():
    a, b = providers.shared_pool(["novita", "zai-org"]), providers.shared_pool(["novita", "zai-org"])
    assert a is b and providers.shared_pool(["zai-org", "novita"]) is not a


def test_pool_is_safe_across_trial_threads():
    pool = providers.Pool(["novita", "zai-org"])
    threads = [threading.Thread(target=lambda: [pool.fallback(pool.first()) for _ in range(30)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert pool.first() in {"novita", "zai-org"} and pool.fallbacks_total == 240
