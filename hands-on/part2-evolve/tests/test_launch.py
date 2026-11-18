"""Laptop side of an evaluation: one HF Job in the attendee's namespace, records read back from its log."""

import pytest

from aidev_evolve import evaljob, launch, transport


class FakeJob:
    def __init__(self, id):
        self.id = id
        self.url = f"https://huggingface.co/jobs/alice/{id}"


class FakeApi:
    """Plays an HF Job: run_job records the spec, fetch_job_logs yields scripted lines (one list per follow)."""

    def __init__(self, follows, stage="COMPLETED"):
        self.follows = list(follows)
        self.submitted = []
        self.stage = stage

    def run_job(self, **spec):
        self.submitted.append(spec)
        return FakeJob("job1")

    def fetch_job_logs(self, job_id, namespace, follow):
        if not self.follows:
            raise AssertionError("unplanned log follow")
        yield from self.follows.pop(0)

    def inspect_job(self, job_id, namespace):
        return type("Info", (), {"status": type("S", (), {"stage": self.stage, "message": "boom"})()})()


def rec(task, outcome="pass"):
    return {"candidate": "c001", "task": task, "outcome": outcome, "cost_usd": 0.01}


def lines_for(*recs, done=True):
    out = ["Installed 120 packages in 3.1s"]
    for i, r in enumerate(recs):
        out += transport.encode(r, msg_id=f"m{i}")
    if done:
        out.append(transport.done_line(len(recs)))
    return out


SPEC = {"candidates": {"c001": "def run(task, env):\n    pass\n"}, "runs": [{"candidate": "c001", "tasks": ["fix-git"], "trials": 1}], "budget_s": 150}


def test_job_spec_carries_the_token_only_as_a_secret():
    js = launch.job_spec("hf_secret", "alice", SPEC, package_url="https://huggingface.co/x/resolve/main/aidev_evolve-0.1.0-py3-none-any.whl")
    assert js["secrets"] == {"HF_TOKEN": "hf_secret"}
    assert "hf_secret" not in str({k: v for k, v in js.items() if k != "secrets"})
    assert js["namespace"] == "alice" and js["flavor"] == "cpu-basic"
    assert evaljob.decode_spec(js["env"][evaljob.SPEC_ENV]) == SPEC
    assert "aidev-evolve eval-job" in js["command"][-1]
    assert js["timeout"] >= 150 + 300


def test_reads_records_live_and_returns_them_with_job_cost():
    api = FakeApi([lines_for(rec("fix-git"), rec("regex-log", "fail"))])
    seen = []
    res = launch.run_eval(api, "hf_x", "alice", SPEC, on_record=seen.append, package_url="u", clock=iter(range(0, 1000, 60)).__next__)
    assert [r["task"] for r in seen] == ["fix-git", "regex-log"]
    assert res.records == seen and res.job_id == "job1"
    assert res.job_usd > 0


def test_a_dropped_log_stream_is_followed_again_without_duplicates():
    all_lines = lines_for(rec("a"), rec("b"))
    api = FakeApi([all_lines[:3], all_lines])  # the first follow drops mid-way
    res = launch.run_eval(api, "hf_x", "alice", SPEC, package_url="u", stage_running=lambda *a: True)
    assert [r["task"] for r in res.records] == ["a", "b"]


def test_a_job_that_dies_without_the_done_line_raises_with_its_log_tail():
    api = FakeApi([["Traceback (most recent call last):", "ModuleNotFoundError: No module named 'hlb'"]], stage="ERROR")
    with pytest.raises(launch.EvalJobError, match="hlb"):
        launch.run_eval(api, "hf_x", "alice", SPEC, package_url="u", stage_running=lambda *a: False)


def test_a_job_that_times_out_after_sending_some_results_returns_them_with_the_gap_counted():
    spec = {**SPEC, "runs": [{"candidate": "c001", "tasks": ["a", "b", "c"], "trials": 1}]}
    api = FakeApi([lines_for(rec("a"), rec("b"), done=False)], stage="ERROR")
    res = launch.run_eval(api, "hf_x", "alice", spec, package_url="u", stage_running=lambda *a: False)
    assert [r["task"] for r in res.records] == ["a", "b"] and res.missing == 1


def test_missing_trials_are_reported_not_silently_dropped():
    api = FakeApi([lines_for(rec("fix-git"))])
    spec = {**SPEC, "runs": [{"candidate": "c001", "tasks": ["fix-git", "regex-log"], "trials": 1}]}
    res = launch.run_eval(api, "hf_x", "alice", spec, package_url="u")
    assert res.missing == 1


def test_an_interrupted_wait_cancels_the_job():
    class Interrupting(FakeApi):
        cancelled = []

        def fetch_job_logs(self, job_id, namespace, follow):
            raise KeyboardInterrupt

        def cancel_job(self, job_id, namespace):
            self.cancelled.append(job_id)

    api = Interrupting([])
    with pytest.raises(KeyboardInterrupt):
        launch.run_eval(api, "hf_x", "alice", SPEC, package_url="u")
    assert api.cancelled == ["job1"]


def test_spec_too_large_for_the_environment_is_refused():
    import random

    rnd = random.Random(0)
    big = {**SPEC, "candidates": {"c001": "x = '" + "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(200_000)) + "'\n"}}
    with pytest.raises(ValueError, match="too large"):
        launch.job_spec("t", "alice", big, package_url="u")


def test_the_wheel_is_a_public_github_release_asset_of_this_version():
    """Attendees need no access to anything private: the wheel is a release asset of the public repo (as in Part 1)."""
    assert launch.release_tag("1.2.3") == "evolve-v1.2.3"
    assert launch.default_package_url("1.2.3") == (
        f"https://github.com/{launch.GITHUB_REPO}/releases/download/evolve-v1.2.3/aidev_evolve-1.2.3-py3-none-any.whl")
    assert launch.GITHUB_REPO == "JoelNiklaus/aidev-harness-workshop"


def test_the_job_downloads_the_wheel_without_sending_the_token_anywhere():
    """The token is for Hugging Face only; the wheel host (GitHub) never sees it."""
    cmd = " ".join(launch.job_spec("hf_secret", "alice", SPEC)["command"])
    assert launch.default_package_url() in cmd
    assert "Authorization" not in cmd and "HF_TOKEN" not in cmd and "hf_secret" not in cmd
