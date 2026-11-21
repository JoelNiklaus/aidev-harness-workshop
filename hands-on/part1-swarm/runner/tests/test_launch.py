"""`aidev-swarm launch`: submit an HF Job that runs the swarm trials on HF, so the laptop can close.

Every external is faked: whoami and the Jobs API (run_job). The pure job spec is tested directly.
"""

import subprocess
import sys

import pytest

from aidev_swarm import __version__, cli, launch

TOKEN = "hf_attendee_token_for_tests"


class FakeJob:
    def __init__(self, id):
        self.id = id


class FakeJobs:
    def __init__(self):
        self.calls = []

    def run_job(self, token, **spec):
        self.calls.append((token, spec))
        return FakeJob("6912abc")


def deps(jobs, can_pay=True, package_ok=True):
    return cli.Deps(whoami=lambda token: {"name": "alice", "canPay": can_pay}, run_job=jobs.run_job,
                    package_status=lambda url: 200 if package_ok else 404)


def run(jobs, *argv, can_pay=True, package_ok=True, env=None):
    return cli.main(["launch", *argv], deps=deps(jobs, can_pay, package_ok), environ={"HF_TOKEN": TOKEN, **(env or {})})


def flat(spec):
    return " ".join(spec["command"])


def test_spec_passes_the_token_only_as_a_job_secret():
    spec = launch.job_spec(TOKEN, "alice", "https://board.example", n=2)
    assert spec["secrets"] == {"HF_TOKEN": TOKEN}
    rest = {k: v for k, v in spec.items() if k != "secrets"}
    assert TOKEN not in repr(rest)


def test_spec_runs_in_the_attendees_namespace_on_cpu_basic_with_the_coordinator_url():
    spec = launch.job_spec(TOKEN, "alice", "https://board.example", n=2)
    assert spec["namespace"] == "alice" and spec["flavor"] == "cpu-basic"
    assert spec["env"]["AIDEV_SWARM_URL"] == "https://board.example"
    assert spec["labels"]["aidev-swarm"] == "runner"


def test_spec_with_n_runs_k_trials_and_its_timeout_grows_with_k():
    two, four = launch.job_spec(TOKEN, "alice", "u", n=2), launch.job_spec(TOKEN, "alice", "u", n=4)
    assert "aidev-swarm run --n 2" in flat(two) and "--loop" not in flat(two)
    assert four["timeout"] > two["timeout"] >= 2 * 40 * 60


def test_spec_runs_the_default_number_of_trials_at_once():
    assert f"--parallel {launch.DEFAULT_PARALLEL}" in flat(launch.job_spec(TOKEN, "alice", "u", n=None))
    assert "--parallel 3" in flat(launch.job_spec(TOKEN, "alice", "u", n=None, parallel=3))


def test_spec_with_n_runs_at_most_n_trials_at_once():
    assert "--n 2 --parallel 2" in flat(launch.job_spec(TOKEN, "alice", "u", n=2))


def test_launch_passes_parallel_to_the_job(capsys):
    jobs = FakeJobs()
    assert run(jobs, "--parallel", "5") == 0
    assert "--parallel 5" in flat(jobs.calls[-1][1])


def test_spec_without_n_loops_until_the_grid_is_full():
    spec = launch.job_spec(TOKEN, "alice", "u", n=None)
    assert "aidev-swarm run --loop" in flat(spec)
    assert spec["timeout"] >= 2 * 3600


def test_spec_installs_the_same_runner_version_with_the_run_extra():
    cmd = flat(launch.job_spec(TOKEN, "alice", "u", n=1))
    assert f"aidev_swarm-{__version__}-py3-none-any.whl" in cmd and "aidev-swarm[run] @ file://" in cmd


def test_package_url_can_be_overridden():
    cmd = flat(launch.job_spec(TOKEN, "alice", "u", n=1, package_url="https://example.org/x/aidev_swarm-9.9.9-py3-none-any.whl"))
    assert "https://example.org/x/aidev_swarm-9.9.9-py3-none-any.whl" in cmd and "aidev_swarm-9.9.9-py3-none-any.whl[" not in cmd


def test_launch_submits_one_job_and_exits_at_once_with_where_to_look(capsys):
    jobs = FakeJobs()
    assert run(jobs, "--n", "2", "--coordinator", "https://board.example") == 0
    [(token, spec)] = jobs.calls
    assert token == TOKEN and spec["namespace"] == "alice" and "--n 2" in flat(spec)
    out = capsys.readouterr()
    assert "6912abc" in out.out and "https://huggingface.co/jobs/alice/6912abc" in out.out and "https://board.example" in out.out
    assert "uvx hf jobs logs -f 6912abc" in out.out and "uvx hf jobs cancel 6912abc" in out.out  # attendees install only uv
    assert TOKEN not in out.out + out.err


def test_launch_uses_the_default_coordinator_when_none_is_given():
    jobs = FakeJobs()
    assert run(jobs) == 0
    assert jobs.calls[0][1]["env"]["AIDEV_SWARM_URL"] == cli.DEFAULT_COORDINATOR


def test_launch_refuses_an_account_without_credits_before_submitting(capsys):
    jobs = FakeJobs()
    assert run(jobs, can_pay=False) == 2
    assert jobs.calls == [] and "credit" in capsys.readouterr().err.lower()


def test_launch_rejects_a_non_positive_n():
    with pytest.raises(SystemExit):
        run(FakeJobs(), "--n", "0")


def test_a_refused_job_submission_is_a_clean_error(capsys):
    jobs = FakeJobs()

    def refuse(token, **spec):
        raise RuntimeError("402 Payment Required")

    jobs.run_job = refuse
    assert run(jobs) == 1
    err = capsys.readouterr().err
    assert "402" in err and TOKEN not in err


def test_launch_works_without_the_heavy_run_extra_installed():
    """The laptop installs only the light package: launch must not import hlb or harbor."""
    code = ("import sys; sys.modules['hlb'] = None; sys.modules['harbor'] = None\n"
            "from aidev_swarm import cli\n"
            "class J: id = 'j1'\n"
            "rc = cli.main(['launch', '--n', '1'], deps=cli.Deps(whoami=lambda t: {'name': 'a', 'canPay': True}, run_job=lambda t, **s: J(), package_status=lambda u: 200),"
            " environ={'HF_TOKEN': 'hf_x'})\n"
            "sys.exit(rc)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr


def test_launch_refuses_when_the_runner_package_cannot_be_downloaded(capsys):
    """Otherwise the Job starts, fails to download the wheel, and the attendee sees only a launched Job."""
    jobs = FakeJobs()
    assert run(jobs, package_ok=False) == 1
    err = capsys.readouterr().err
    assert jobs.calls == [] and "404" in err and launch.default_package_url() in err


def test_the_runner_wheel_is_a_public_github_release_asset_of_this_version():
    """Attendees need no access to anything private: the wheel is a release asset of the public repo."""
    assert launch.default_package_url("1.2.3") == (
        f"https://github.com/{launch.GITHUB_REPO}/releases/download/{launch.release_tag('1.2.3')}/aidev_swarm-1.2.3-py3-none-any.whl")
    assert launch.release_tag("1.2.3") == "swarm-v1.2.3"


def test_the_job_downloads_the_wheel_without_sending_the_token_anywhere():
    """The token is for Hugging Face only; the wheel host (GitHub) never sees it."""
    cmd = flat(launch.job_spec(TOKEN, "alice", "u", n=1))
    assert "Authorization" not in cmd and "HF_TOKEN" not in cmd


def test_the_package_check_sends_no_credentials(monkeypatch):
    import httpx

    seen = {}

    def head(url, **kw):
        seen.update(kw, url=url)
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(httpx, "head", head)
    assert launch.package_status("https://github.com/x/y/releases/download/t/w.whl") == 200
    assert "Authorization" not in repr(seen.get("headers") or {}) and "auth" not in seen
