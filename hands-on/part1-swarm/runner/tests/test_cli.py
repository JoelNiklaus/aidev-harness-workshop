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

    def release(self, cell_id, quick=False):
        self.log.append(("release", cell_id))
        self.quick_releases = getattr(self, "quick_releases", []) + ([cell_id] if quick else [])
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
        self.providers = []  # the provider each trial was asked to use

    def deps(self):
        def run_trial(claim, grid, token, user, workdir):
            self.trials.append((claim["cell_id"], token, user))
            self.providers.append(grid.get("provider"))
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
    for name in ("grid.json", "prices.json", "providers.json"):
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


# ---------------------------------------------------------------- dead jobs give their cells back fast (Joel, 18 Nov)


def test_the_runner_heartbeats_every_30_s_with_jitter_so_500_runners_do_not_beat_in_step():
    import random

    assert cli.HEARTBEAT_S == 30 and cli.Deps().heartbeat_s == cli.HEARTBEAT_S
    waits = [cli.beat_wait(30, random.Random(k)) for k in range(200)]
    assert 24 <= min(waits) and max(waits) <= 30 and max(waits) - min(waits) > 3


def test_an_interrupt_during_the_submit_still_gives_the_cell_back():
    h = Harness([cell()])

    def cancelled(cell_id, result):
        raise KeyboardInterrupt  # SIGTERM from `hf jobs cancel` while the result is on its way

    h.coord.submit = cancelled
    assert main(h, "run", "--coordinator", "https://c") == 130
    assert ("release", "opencode|nginx-request-logging|0") in h.coord.log


def test_any_other_exit_from_the_trial_gives_the_cell_back_quickly():
    def exits(claim, workdir):
        raise SystemExit(3)

    h = Harness([cell()], trial=exits)
    with pytest.raises(SystemExit):
        main(h, "run", "--coordinator", "https://c")
    assert h.coord.quick_releases == ["opencode|nginx-request-logging|0"]


def test_the_exit_hook_releases_the_cells_held_at_that_moment(monkeypatch):
    hooks = []
    monkeypatch.setattr(cli.atexit, "register", lambda fn, *a: hooks.append((fn, a)))
    monkeypatch.setattr(cli.atexit, "unregister", lambda fn: None)
    h = Harness([cell()])

    def interpreter_exits(claim, workdir):
        fn, a = hooks[-1]
        fn(*a)  # what the interpreter runs at exit while this cell is held
        return copy_trial(workdir)

    h._trial = interpreter_exits
    main(h, "run", "--coordinator", "https://c")
    assert h.coord.log[:2] == [("claim", 1), ("release", "opencode|nginx-request-logging|0")]
    assert h.coord.quick_releases == ["opencode|nginx-request-logging|0"]


def test_a_second_sigterm_does_not_interrupt_the_release():
    import os
    import signal

    h = Harness([cell()])
    plain = h.coord.release

    def release(cell_id, quick=False):
        os.kill(os.getpid(), signal.SIGTERM)  # the Job runtime signals again
        plain(cell_id, quick)

    def cancelled(claim, workdir):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM did not interrupt the trial")

    h.coord.release = release
    h._trial = cancelled
    before = signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(AssertionError("outer handler ran")))
    try:
        assert main(h, "run", "--coordinator", "https://c") == 130
        assert ("release", "opencode|nginx-request-logging|0") in h.coord.log
    finally:
        signal.signal(signal.SIGTERM, before)


def test_when_every_free_cell_is_taken_the_runner_waits_and_takes_one_that_comes_back(capsys):
    from aidev_swarm.coordinator import AllTaken

    h = Harness([cell(0)])
    plain, sleeps, calls = h.coord.claim, [], []

    def busy_then_free(n=1):
        calls.append(n)
        if len(calls) <= 2:
            raise AllTaken(30)
        return plain(n)

    h.coord.claim = busy_then_free
    deps = h.deps()
    deps.sleep = sleeps.append
    rc = cli.main(["run", "--coordinator", "https://c"], deps=deps, environ={"HF_TOKEN": TOKEN, "AIDEV_SWARM_WORKDIR": str(WORKDIR)})
    assert rc == 0 and sleeps == [30, 30] and list(h.coord.submitted) == ["opencode|nginx-request-logging|0"]
    assert capsys.readouterr().out.count("waiting") == 1  # said once, not every 30 s


def test_a_lost_lease_is_reported_by_the_heartbeat(capsys):
    from aidev_swarm.coordinator import LeaseLost

    h = Harness([cell()])
    seen = threading.Event()

    def lost(cell_id):
        seen.set()
        raise LeaseLost("/api/heartbeat/x: the lease belongs to someone else")

    h.coord.heartbeat = lost

    def slow(claim, workdir):
        assert seen.wait(5)
        return copy_trial(workdir)

    h._trial = slow
    main(h, "run", "--coordinator", "https://c")
    assert "lease lost" in capsys.readouterr().out


# ---- provider spread (config/providers.json): one provider per attendee, the next one after provider trouble


def provider_trouble(n_errors, *, infra=False):
    """A finished trial whose meter saw `n_errors` 429/5xx answers (or, with infra, no model call at all)."""
    import json

    def trial(claim, workdir):
        d = copy_trial(workdir)
        log = d / "agent" / "hlb-meter.jsonl"
        if infra:
            res = json.loads((d / "result.json").read_text())
            res["verifier_result"], res["exception_info"] = None, {"exception_type": "AgentError", "exception_message": "429 from upstream"}
            (d / "result.json").write_text(json.dumps(res))
            log.write_text("")
        else:
            log.write_text(log.read_text() + "".join(json.dumps({"path": "/v1/chat/completions", "status": 429}) + "\n" for _ in range(n_errors)))
        return d

    return trial


def test_each_trial_uses_the_attendees_provider_and_the_result_records_it():
    from aidev_swarm import providers

    first = providers.order("alice", providers.load())[0]
    h = Harness([cell(0), cell(1)])
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 0
    assert h.providers == [first, first]
    assert {r["provider"] for r in h.coord.submitted.values()} == {first}
    assert all(r["n_provider_errors"] == 0 for r in h.coord.submitted.values())


def test_a_trial_with_provider_errors_moves_the_next_trial_to_the_next_provider(capsys):
    from aidev_swarm import providers

    order = providers.order("alice", providers.load())
    calls = iter([provider_trouble(cli.PROVIDER_ERRORS_TO_SWITCH), provider_trouble(0)])
    h = Harness([cell(0), cell(1)], trial=lambda claim, workdir: next(calls)(claim, workdir))
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 0
    assert h.providers == [order[0], order[1]]
    assert h.coord.submitted["opencode|nginx-request-logging|0"]["n_provider_errors"] == cli.PROVIDER_ERRORS_TO_SWITCH
    assert f"switching to {order[1]}" in capsys.readouterr().out


def test_a_few_provider_errors_keep_the_provider():
    calls = iter([provider_trouble(cli.PROVIDER_ERRORS_TO_SWITCH - 1), provider_trouble(0)])
    h = Harness([cell(0), cell(1)], trial=lambda claim, workdir: next(calls)(claim, workdir))
    assert main(h, "run", "--coordinator", "https://c", "--loop") == 0
    assert h.providers[0] == h.providers[1]


def test_an_infra_failure_before_any_model_call_also_moves_to_the_next_provider():
    from aidev_swarm import providers

    order = providers.order("alice", providers.load())
    calls = iter([provider_trouble(0, infra=True), provider_trouble(0)])
    h = Harness([cell(0), cell(1)], trial=lambda claim, workdir: next(calls)(claim, workdir))
    main(h, "run", "--coordinator", "https://c", "--loop")
    assert h.providers == [order[0], order[1]]


def test_the_coordinator_can_still_force_a_provider_through_run_config():
    c = cell(0)
    c["run_config"] = {"provider": "zai-org"}
    h = Harness([c])
    assert main(h, "run", "--coordinator", "https://c") == 0
    assert h.coord.submitted[c["cell_id"]]["provider"] == "zai-org"
