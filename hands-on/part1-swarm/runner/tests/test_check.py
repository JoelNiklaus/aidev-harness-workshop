"""`aidev-swarm check`: the prep email's self-test. One model call and one tiny HF Job, no cell claimed.

Every external is faked: whoami, the inference router, the Jobs API and the clock.
"""

import subprocess
import sys

from aidev_swarm import check, cli

TOKEN = "hf_attendee_token_for_tests"


class FakeJob:
    def __init__(self, id):
        self.id = id


class Fakes:
    def __init__(self, reply="ok", stages=("RUNNING", "COMPLETED"), refuse_job=None, refuse_chat=None, refuse_sandbox=None):
        self.chats, self.jobs, self.cancelled, self.slept, self.sandboxes = [], [], [], 0, []
        self.refuse_sandbox = refuse_sandbox
        self.reply, self.stages, self.refuse_job, self.refuse_chat = reply, list(stages), refuse_job, refuse_chat

    def chat(self, token, model, provider):
        self.chats.append((token, model, provider))
        if self.refuse_chat:
            raise RuntimeError(self.refuse_chat)
        return self.reply

    def run_job(self, token, **spec):
        self.jobs.append((token, spec))
        if self.refuse_job:
            raise RuntimeError(self.refuse_job)
        return FakeJob("6912chk")

    def sandbox_echo(self, token, user):
        self.sandboxes.append((token, user))
        if self.refuse_sandbox:
            raise RuntimeError(self.refuse_sandbox)
        return "sandbox ok"

    def job_stage(self, token, user, job_id):
        return self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]

    def cancel_job(self, token, user, job_id):
        self.cancelled.append(job_id)

    def sleep(self, s):
        self.slept += 1
        if self.slept > 1000:  # a missing exit must fail fast, never hang the suite
            raise RuntimeError("fake clock: too many sleeps")

    def coordinator(self, url, token):
        raise AssertionError("check must not talk to the coordinator")

    def deps(self, can_pay=True):
        return cli.Deps(whoami=lambda t: {"name": "alice", "canPay": can_pay}, chat=self.chat, run_job=self.run_job,
                        job_stage=self.job_stage, sandbox_echo=self.sandbox_echo, cancel_job=self.cancel_job, sleep=self.sleep, coordinator=self.coordinator,
                        check_poll_s=3, check_wait_s=30)


def run(f, can_pay=True):
    return cli.main(["check"], deps=f.deps(can_pay), environ={"HF_TOKEN": TOKEN})


def test_check_calls_the_workshop_model_once_runs_a_tiny_job_and_reports_ready(capsys):
    f = Fakes()
    assert run(f) == 0
    grid = check.grid()
    assert f.chats == [(TOKEN, grid["model"], grid["provider"])]
    [(token, spec)] = f.jobs
    assert token == TOKEN and spec["namespace"] == "alice" and spec["flavor"] == "cpu-basic"
    assert spec["timeout"] <= 600 and spec["labels"] == {"aidev-swarm": "check"}
    out = capsys.readouterr()
    assert "https://huggingface.co/jobs/alice/6912chk" in out.out and "ready" in out.out.lower()
    assert TOKEN not in out.out + out.err


def test_check_runs_one_command_in_an_hf_sandbox_like_a_trial_does(capsys):
    f = Fakes()
    assert run(f) == 0
    assert f.sandboxes == [(TOKEN, "alice")]
    assert "sandbox ok" in capsys.readouterr().out


def test_a_failed_sandbox_fails_with_the_jobs_hint(capsys):
    f = Fakes(refuse_sandbox="403 Forbidden")
    assert run(f) == 1
    err = capsys.readouterr().err
    assert "sandbox" in err.lower() and "403" in err and "Jobs" in err and TOKEN not in err


def test_the_check_job_never_receives_the_token():
    spec = check.job_spec("alice")
    assert "secrets" not in spec and TOKEN not in repr(spec)


def test_check_claims_nothing():
    f = Fakes()
    assert run(f) == 0  # the fake coordinator raises if touched


def test_a_refused_model_call_fails_with_the_permission_hint_and_still_tests_jobs(capsys):
    f = Fakes(refuse_chat="403 Forbidden")
    assert run(f) == 1
    err = capsys.readouterr().err
    assert "403" in err and "Inference Providers" in err and TOKEN not in err
    assert len(f.jobs) == 1, "report every problem in one run"


def test_a_refused_job_fails_with_the_jobs_hint(capsys):
    f = Fakes(refuse_job="402 Payment Required")
    assert run(f) == 1
    err = capsys.readouterr().err
    assert "402" in err and "Jobs" in err and TOKEN not in err


def test_a_job_that_ends_in_error_fails(capsys):
    f = Fakes(stages=("RUNNING", "ERROR"))
    assert run(f) == 1
    assert "ERROR" in capsys.readouterr().err


def test_a_job_that_does_not_finish_in_time_is_cancelled_and_fails(capsys):
    f = Fakes(stages=("SCHEDULING",))
    assert run(f) == 1
    assert f.cancelled == ["6912chk"]
    assert "6912chk" in capsys.readouterr().err


def test_an_account_without_credits_is_refused_before_any_call(capsys):
    f = Fakes()
    assert run(f, can_pay=False) == 2
    assert f.chats == [] and f.jobs == [] and "credit" in capsys.readouterr().err.lower()


def test_check_works_without_the_heavy_run_extra_installed():
    """The prep email runs check on the light laptop install: it must not import hlb or harbor."""
    code = ("import sys; sys.modules['hlb'] = None; sys.modules['harbor'] = None\n"
            "from aidev_swarm import cli\n"
            "class J: id = 'j1'\n"
            "rc = cli.main(['check'], deps=cli.Deps(whoami=lambda t: {'name': 'a', 'canPay': True}, chat=lambda t, m, p: 'ok',"
            " run_job=lambda t, **s: J(), job_stage=lambda t, u, j: 'COMPLETED', sandbox_echo=lambda t, u: 'sandbox ok', sleep=lambda s: None), environ={'HF_TOKEN': 'hf_x'})\n"
            "sys.exit(rc)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
