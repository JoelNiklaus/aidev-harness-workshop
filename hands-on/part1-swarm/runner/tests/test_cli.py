"""`aidev-swarm run` end to end with every external faked: whoami, the coordinator (in-memory), Harbor."""

import shutil
import threading
from pathlib import Path

import pytest

from aidev_swarm import cli

FIX = Path(__file__).parent / "fixtures" / "trial_pass"
TOKEN = "hf_attendee_token_for_tests"


class FakeCoordinator:
    def __init__(self, cells):
        self.cells = list(cells)
        self.log = []
        self.submitted = {}
        self.heartbeat_seen = threading.Event()
        self.held = set()

    def claim(self, n=1):
        """Like the real coordinator: `n` caps the user's live leases; n=2 answers with a list."""
        from aidev_swarm.coordinator import AtLimit

        self.log.append(("claim", n))
        if len(self.held) >= n:
            raise AtLimit("at limit", sorted(self.held))
        if not self.cells:
            return None if n == 1 else []
        c = self.cells.pop(0)
        self.held.add(c["cell_id"])
        return c if n == 1 else [c]

    def heartbeat(self, cell_id):
        self.log.append(("heartbeat", cell_id))
        self.heartbeat_seen.set()

    def submit(self, cell_id, result):
        self.log.append(("submit", cell_id))
        self.submitted[cell_id] = result
        self.held.discard(cell_id)

    def release(self, cell_id):
        self.log.append(("release", cell_id))
        self.held.discard(cell_id)


def cell(i=0, harness="opencode", task="nginx-request-logging"):
    return {"cell_id": f"{harness}|{task}|{i}", "harness": harness, "task": task, "trial": i,
            "model": "zai-org/GLM-5.3-Flash", "lease_expires_at": "2026-11-17T12:00:00Z", "run_config": {}}


def copy_trial(workdir):
    import uuid

    d = Path(workdir) / f"job-{uuid.uuid4().hex[:6]}" / "nginx-request-logging__x"
    shutil.copytree(FIX, d)
    return d


class Harness:
    """Wires fakes into cli.main and records what happened."""

    def __init__(self, cells=(), can_pay=True, trial=None):
        self.coord = FakeCoordinator(cells)
        self.trials = []
        self.can_pay = can_pay
        self._trial = trial

    def deps(self):
        def run_trial(claim, grid, token, user, workdir):
            self.trials.append((claim["cell_id"], token, user))
            assert Path(workdir) == WORKDIR
            return (self._trial or (lambda *a: copy_trial(workdir)))(claim, workdir)

        return cli.Deps(whoami=lambda token: {"name": "alice", "canPay": self.can_pay},
                        coordinator=lambda url, token: self.coord, run_trial=run_trial,
                        sandbox_id=lambda cell_id, user: "sbx123", heartbeat_s=0.01)


@pytest.fixture(autouse=True)
def workdir(tmp_path):
    global WORKDIR
    WORKDIR = tmp_path / "jobs"
    yield WORKDIR


def main(h, *argv, env=None):
    return cli.main(list(argv), deps=h.deps(), environ={"HF_TOKEN": TOKEN, "AIDEV_SWARM_WORKDIR": str(WORKDIR), **(env or {})})


def test_no_token_exits_with_a_hint(capsys):
    h = Harness([cell()])
    rc = cli.main(["run", "--coordinator", "https://c"], deps=h.deps(), environ={}, token_lookup=lambda: None)
    assert rc == 2 and "hf auth login" in capsys.readouterr().err
    assert h.coord.log == []


def test_account_without_credits_is_refused_before_claiming(capsys):
    h = Harness([cell()], can_pay=False)
    assert main(h, "run", "--coordinator", "https://c") == 2
    assert "credit" in capsys.readouterr().err.lower()
    assert h.coord.log == []


def test_dry_claims_nothing(capsys):
    h = Harness([cell()])
    assert main(h, "run", "--coordinator", "https://c", "--dry") == 0
    out = capsys.readouterr().out
    assert "alice" in out and h.coord.log == [] and h.trials == []


def test_run_claims_runs_and_submits_the_protocol_result(capsys):
    h = Harness([cell()])
    assert main(h, "run", "--coordinator", "https://swarm.example") == 0
    r = h.coord.submitted["opencode|nginx-request-logging|0"]
    assert r["passed"] is True and r["hf_user"] == "alice" and r["sandbox_id"] == "sbx123" and r["runner_version"] == cli.__version__
    assert r["cost_usd"] > 0 and r["harness"] == "opencode"
    assert h.trials == [("opencode|nginx-request-logging|0", TOKEN, "alice")]
    out = capsys.readouterr()
    assert "https://swarm.example" in out.out and TOKEN not in out.out + out.err


def test_full_grid_ends_cleanly_without_a_trial(capsys):
    h = Harness([])
    assert main(h, "run", "--coordinator", "https://c") == 0
    assert h.trials == [] and "nothing left" in capsys.readouterr().out.lower()


def test_coordinator_url_can_come_from_the_environment():
    h = Harness([cell()])
    assert main(h, "run", env={"AIDEV_SWARM_URL": "https://from-env"}) == 0
    assert h.coord.submitted


def test_without_a_coordinator_url_the_workshop_space_is_used():
    h = Harness([cell()])
    urls = []
    deps = h.deps()
    deps.coordinator = lambda url, token: urls.append(url) or h.coord
    assert cli.main(["run"], deps=deps, environ={"HF_TOKEN": TOKEN, "AIDEV_SWARM_WORKDIR": str(WORKDIR)}) == 0
    assert urls == [cli.DEFAULT_COORDINATOR] == ["https://agent-harness-workshop-aidev-swarm.hf.space"]


def test_crash_in_the_trial_releases_the_cell():
    def boom(claim, workdir):
        raise RuntimeError("sandbox died")

    h = Harness([cell()], trial=boom)
    assert main(h, "run", "--coordinator", "https://c") == 1
    assert ("release", "opencode|nginx-request-logging|0") in h.coord.log and not h.coord.submitted


def test_ctrl_c_releases_the_cell():
    def interrupted(claim, workdir):
        raise KeyboardInterrupt

    h = Harness([cell()], trial=interrupted)
    assert main(h, "run", "--coordinator", "https://c") == 130
    assert ("release", "opencode|nginx-request-logging|0") in h.coord.log


def test_infra_failure_before_any_model_call_releases_instead_of_submitting(tmp_path):
    import json

    def infra(claim, workdir):
        d = copy_trial(workdir)
        res = json.loads((d / "result.json").read_text())
        res["verifier_result"], res["exception_info"] = None, {"exception_type": "EnvironmentStartTimeoutError", "exception_message": "pull"}
        (d / "result.json").write_text(json.dumps(res))
        (d / "agent" / "hlb-meter.jsonl").write_text("")
        return d

    h = Harness([cell()], trial=infra)
    assert main(h, "run", "--coordinator", "https://c") == 1
    assert ("release", "opencode|nginx-request-logging|0") in h.coord.log and not h.coord.submitted


def test_heartbeats_are_sent_while_the_trial_runs():
    h = Harness([cell()])

    def slow(claim, workdir):
        assert h.coord.heartbeat_seen.wait(5), "no heartbeat while the trial ran"
        return copy_trial(workdir)

    h._trial = slow
    assert main(h, "run", "--coordinator", "https://c") == 0
    assert ("heartbeat", "opencode|nginx-request-logging|0") in h.coord.log


def test_loop_runs_cells_until_the_grid_is_full():
    h = Harness([cell(0), cell(1)])
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 0
    assert sorted(h.coord.submitted) == ["opencode|nginx-request-logging|0", "opencode|nginx-request-logging|1"]
    assert h.coord.log[-1] == ("claim", 1)


def test_local_mode_runs_one_trial_without_a_coordinator(capsys):
    h = Harness([])
    assert main(h, "run", "--local", "--harness", "pi", "--task", "nginx-request-logging") == 0
    assert h.coord.log == [] and h.trials[0][0] == "pi__nginx-request-logging__0"
    assert '"passed": true' in capsys.readouterr().out


def test_packaged_config_matches_the_repo_config():
    repo = Path(__file__).parents[2] / "config"
    for name in ("grid.json", "prices.json"):
        assert (repo / name).read_text() == cli.data_text(name), f"runner/src/aidev_swarm/data/{name} is stale: copy config/{name}"


def test_rejected_submit_keeps_the_result_on_screen_and_exits_nonzero(capsys):
    from aidev_swarm.coordinator import LeaseLost

    h = Harness([cell()])

    def refuse(cell_id, result):
        raise LeaseLost("lease held by another user")

    h.coord.submit = refuse
    assert main(h, "run", "--coordinator", "https://c") == 1
    out = capsys.readouterr().out
    assert "not accepted" in out and '"cell_id": "opencode|nginx-request-logging|0"' in out


def test_unreachable_coordinator_on_claim_is_a_clean_error(capsys):
    from aidev_swarm.coordinator import CoordinatorError

    h = Harness([cell()])

    def down(n=1):
        raise CoordinatorError("/api/claim: gave up after 6 attempts (ConnectError)")

    h.coord.claim = down
    assert main(h, "run", "--coordinator", "https://c") == 1
    assert "coordinator" in capsys.readouterr().err.lower() and h.trials == []


def test_invalid_token_is_a_clean_error(capsys):
    h = Harness([cell()])
    deps = h.deps()

    def bad(token):
        raise PermissionError("401")

    deps.whoami = bad
    rc = cli.main(["run", "--coordinator", "https://c"], deps=deps, environ={"HF_TOKEN": TOKEN})
    err = capsys.readouterr().err
    assert rc == 2 and "token" in err.lower() and TOKEN not in err


def test_local_mode_does_not_say_claimed(capsys):
    h = Harness([])
    main(h, "run", "--local", "--harness", "pi", "--task", "nginx-request-logging")
    assert "claimed" not in capsys.readouterr().out


def test_release_failing_after_a_crash_still_ends_cleanly(capsys):
    from aidev_swarm.coordinator import CoordinatorError

    def boom(claim, workdir):
        raise RuntimeError("sandbox died")

    h = Harness([cell()], trial=boom)

    def down(cell_id):
        raise CoordinatorError("/api/release: gave up after 6 attempts (ConnectError)")

    h.coord.release = down
    assert main(h, "run", "--coordinator", "https://c") == 1
    out = capsys.readouterr().out
    assert "sandbox died" in out and "lease expires" in out


def test_local_mode_uses_the_coordinator_cell_id_format(capsys):
    h = Harness([])
    main(h, "run", "--local", "--harness", "pi", "--task", "nginx-request-logging")
    assert h.trials[0][0] == "pi__nginx-request-logging__0"


def failing_for(*cell_ids):
    """A trial function whose setup fails (crash before any model call) for the given cells."""
    def trial(claim, workdir):
        if claim["cell_id"] in cell_ids:
            raise RuntimeError("agent setup failed")
        return copy_trial(workdir)

    return trial


def test_n_runs_at_most_k_trials():
    h = Harness([cell(0), cell(1), cell(2)])
    assert main(h, "run", "--coordinator", "https://c", "--n", "2") == 0
    assert len(h.trials) == 2 and len(h.coord.submitted) == 2 and len(h.coord.cells) == 1


def test_loop_skips_to_the_next_cell_after_a_setup_failure():
    h = Harness([cell(0), cell(1)], trial=failing_for("opencode|nginx-request-logging|0"))
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 0
    assert list(h.coord.submitted) == ["opencode|nginx-request-logging|1"]
    log = h.coord.log
    # the failed cell is held while the next one is claimed (n=2), so the queue cannot hand it straight back
    assert log.index(("claim", 2)) < log.index(("release", "opencode|nginx-request-logging|0"))


def test_loop_stops_after_three_consecutive_setup_failures(capsys):
    ids = [f"opencode|nginx-request-logging|{i}" for i in range(5)]
    h = Harness([cell(i) for i in range(5)], trial=failing_for(*ids))
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 1
    assert len(h.trials) == 3 and not h.coord.submitted
    assert all(("release", i) in h.coord.log for i in ids[:3]) and not h.coord.held
    assert "3 trials in a row" in capsys.readouterr().out


def test_a_success_resets_the_consecutive_failure_count():
    bad = [f"opencode|nginx-request-logging|{i}" for i in (0, 1, 3, 4)]
    h = Harness([cell(i) for i in range(6)], trial=failing_for(*bad))
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 0
    assert len(h.trials) == 6 and sorted(h.coord.submitted) == ["opencode|nginx-request-logging|2", "opencode|nginx-request-logging|5"]


def test_without_loop_a_setup_failure_releases_at_once_and_stops():
    h = Harness([cell(0), cell(1)], trial=failing_for("opencode|nginx-request-logging|0"))
    assert main(h, "run", "--coordinator", "https://c") == 1
    assert h.coord.log == [("claim", 1), ("release", "opencode|nginx-request-logging|0")]


def test_a_second_runner_of_the_same_user_claims_its_cell_with_n2():
    h = Harness([cell(0)])
    h.coord.held.add("pi|other-task|0")  # another Job of the same attendee holds one lease
    assert main(h, "run", "--coordinator", "https://c") == 0
    assert h.coord.log[:2] == [("claim", 1), ("claim", 2)] and h.coord.submitted


def test_at_the_lease_limit_even_with_n2_the_runner_stops_with_a_clear_message(capsys):
    h = Harness([cell(0)])
    h.coord.held.update({"pi|a|0", "pi|b|0"})
    assert main(h, "run", "--coordinator", "https://c") == 1
    assert "already" in capsys.readouterr().err and h.trials == []


def test_sigterm_from_a_cancelled_job_releases_the_cell():
    import os
    import signal

    def cancelled(claim, workdir):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM did not interrupt the trial")

    class NotHandled(Exception):
        pass

    def outer(signum, frame):  # stands in for the default action (which would kill pytest)
        raise NotHandled

    before = signal.signal(signal.SIGTERM, outer)
    try:
        h = Harness([cell()], trial=cancelled)
        assert main(h, "run", "--coordinator", "https://c") == 130
        assert ("release", "opencode|nginx-request-logging|0") in h.coord.log
        assert signal.getsignal(signal.SIGTERM) is outer, "run must restore the SIGTERM handler"
    finally:
        signal.signal(signal.SIGTERM, before)


def test_an_unsent_result_is_kept_on_disk_and_resubmitted_by_the_next_run():
    from aidev_swarm.coordinator import CoordinatorError

    h = Harness([cell(0)])
    real_submit = h.coord.submit

    def down(cell_id, result):
        raise CoordinatorError("/api/submit: gave up after 40 attempts (HTTP 429)")

    h.coord.submit = down
    assert main(h, "run", "--coordinator", "https://c") == 1
    unsent = list((WORKDIR / "unsent").glob("*.json"))
    assert len(unsent) == 1
    h.coord.submit = real_submit
    assert main(h, "run", "--coordinator", "https://c") == 0
    assert "opencode|nginx-request-logging|0" in h.coord.submitted and not list((WORKDIR / "unsent").glob("*.json"))


def test_an_unsent_result_whose_lease_was_taken_over_is_dropped(capsys):
    from aidev_swarm.coordinator import CoordinatorError, LeaseLost

    h = Harness([cell(0)])

    def down(cell_id, result):
        raise CoordinatorError("/api/submit: gave up (HTTP 429)")

    h.coord.submit = down
    main(h, "run", "--coordinator", "https://c")

    def taken(cell_id, result):
        raise LeaseLost("/api/submit/x: someone else has it")

    h.coord.submit = taken
    main(h, "run", "--coordinator", "https://c")
    assert not list((WORKDIR / "unsent").glob("*.json"))


def test_an_extra_cell_from_claim_two_is_released_at_once():
    h = Harness([cell(0), cell(1), cell(2)], trial=failing_for("opencode|nginx-request-logging|0"))
    plain = h.coord.claim

    def generous(n=1):  # a coordinator that hands out two new cells for ?n=2
        if n == 2:
            h.coord.log.append(("claim", 2))
            return [h.coord.cells.pop(0), h.coord.cells.pop(0)]
        return plain(n)

    h.coord.claim = generous
    assert main(h, "run", "--coordinator", "https://c", "--n", "2") == 0
    assert ("release", "opencode|nginx-request-logging|2") in h.coord.log
    assert list(h.coord.submitted) == ["opencode|nginx-request-logging|1"]


def test_a_held_failed_cell_is_released_when_the_next_claim_fails():
    from aidev_swarm.coordinator import CoordinatorError

    h = Harness([cell(0), cell(1)], trial=failing_for("opencode|nginx-request-logging|0"))
    plain = h.coord.claim
    calls = []

    def flaky(n=1):
        calls.append(n)
        if len(calls) > 1:
            raise CoordinatorError("/api/claim: gave up (HTTP 503)")
        return plain(n)

    h.coord.claim = flaky
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 1
    assert ("release", "opencode|nginx-request-logging|0") in h.coord.log
