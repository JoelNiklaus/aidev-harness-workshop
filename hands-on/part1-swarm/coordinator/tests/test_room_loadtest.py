"""scripts/room_loadtest.py: the Job specs it submits (the token only as a secret, one load-test name per Job, the
published wheels), the arrival plan, a refused submission, and the `LT` lines it reads back."""

from __future__ import annotations

import ast
import base64
import importlib.util
from pathlib import Path

import pytest

from swarm_coordinator import join
from swarm_coordinator.app import LOADTEST_HEADER, LOADTEST_NAME

spec = importlib.util.spec_from_file_location("room_loadtest", Path(__file__).parents[1] / "scripts" / "room_loadtest.py")
lt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lt)

KW = {"token": "hf_dummy_token_value", "user": "joel", "board": lt.STAGING, "tag": "t1"}


@pytest.mark.parametrize(("kind", "package"), [("part1", "aidev-swarm[run] @ file:///tmp/aidev_swarm-9.9.9-py3-none-any.whl"),
                                               ("part2", "aidev-evolve @ file:///tmp/aidev_swarm-9.9.9-py3-none-any.whl")])
def test_a_job_installs_the_given_wheel_and_keeps_the_token_in_its_secret_only(kind, package):
    s = lt.job_spec(kind, "lt-t1-03", wheel="https://example.org/x/aidev_swarm-9.9.9-py3-none-any.whl", **KW)
    assert s["secrets"] == {"HF_TOKEN": "hf_dummy_token_value"} and s["namespace"] == "joel"
    assert "hf_dummy_token_value" not in repr({k: v for k, v in s.items() if k != "secrets"})
    assert package in s["command"][2] and "curl -fsSL" in s["command"][2]
    assert s["env"]["LT_NAME"] == "lt-t1-03" and s["env"]["LT_HEADER"] == LOADTEST_HEADER and s["env"]["LT_BOARD"] == lt.STAGING
    assert s["labels"][lt.LABEL] == "t1"
    ast.parse(base64.b64decode(s["env"]["LT_DRIVER"]).decode())  # the driver the Job runs is valid Python


def test_the_drivers_change_only_identity_provider_and_measurement():
    """They call the packages' own entry points; nothing of the trial or the loop is replaced."""
    assert 'cli.main(["run", "--loop", "--parallel"' in lt.DRIVER_PART1 and "cli._run_trial(claim, grid, token, user, workdir)" in lt.DRIVER_PART1
    assert 'cli.main(["run", "--block"' in lt.DRIVER_PART2


def test_names_are_load_test_names_the_coordinator_accepts():
    got = lt.names("p1a", 40)
    assert len(set(got)) == 40 and all(LOADTEST_NAME.match(n) for n in got)
    with pytest.raises(SystemExit):
        lt.names("Bad_Tag", 2)


def test_the_plan_spreads_arrivals_and_takes_the_seating_blocks_in_turn():
    p = lt.plan("part2", "t1", 7, arrive_s=180, wheel="https://example.org/a.whl", **{k: v for k, v in KW.items() if k != "tag"})
    assert [round(at) for at, _ in p] == [0, 30, 60, 90, 120, 150, 180]
    assert [s["env"]["LT_BLOCK"] for _, s in p][:4] == [*join.BLOCKS, list(join.BLOCKS)[0]]
    assert lt.plan("part1", "t1", 1, arrive_s=180, wheel="https://example.org/a.whl", **{k: v for k, v in KW.items() if k != "tag"})[0][0] == 0


def test_the_wheels_are_the_ones_the_start_pages_show():
    urls = lt.wheel_urls()
    assert "/swarm-v" in urls["part1"] and "/evolve-v" in urls["part2"] and all(u.endswith(".whl") for u in urls.values())


class FakeJobs:
    def __init__(self, refuse=()):
        self.refuse, self.started = set(refuse), []

    def run_job(self, **spec):
        name = spec["env"]["LT_NAME"]
        if name in self.refuse:
            raise RuntimeError("429 Too Many Requests")
        self.started.append(name)
        return type("J", (), {"id": f"job-{name}"})()


def test_submit_waits_for_each_arrival_and_goes_on_after_a_refused_job():
    now, sleeps, said = [0.0], [], []

    def sleep(s):
        sleeps.append(round(s))
        now[0] += s

    p = lt.plan("part1", "t1", 3, arrive_s=60, wheel="https://example.org/a.whl", **{k: v for k, v in KW.items() if k != "tag"})
    api = FakeJobs(refuse={"lt-t1-01"})
    out = lt.submit(api, p, sleep=sleep, clock=lambda: now[0], say=said.append)
    assert sleeps == [30, 30] and api.started == ["lt-t1-00", "lt-t1-02"]
    assert [d["job"] for d in out] == ["job-lt-t1-00", None, "job-lt-t1-02"] and "429" in out[1]["error"] and said


def test_lt_lines_reads_the_measurements_out_of_a_job_log():
    log = ['[aidev-swarm 10:00:00] claimed pi on fix-git', 'LT {"name": "lt-t1-00", "k": "trial", "s": 99.5}', "LT {broken", 'x LT {"k": "end", "rc": 0}']
    assert lt.lt_lines(iter(log)) == [{"name": "lt-t1-00", "k": "trial", "s": 99.5}, {"k": "end", "rc": 0}]
