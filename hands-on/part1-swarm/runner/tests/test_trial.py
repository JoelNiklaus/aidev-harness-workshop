"""One trial on HF Sandbox through the Harness Leaderboard's Harbor wiring, with Harbor itself faked."""

import json
import os
import shutil
from pathlib import Path

import pytest
from hlb import config as hlb_config

from aidev_swarm import trial as T

FIX = Path(__file__).parent / "fixtures" / "trial_pass"
TOKEN = "hf_attendee_token_for_tests"
MODEL = "zai-org/GLM-5.3-Flash"


@pytest.fixture(autouse=True)
def restore_owner():
    owner = hlb_config.OWNER
    yield
    hlb_config.OWNER = owner


def test_run_on_the_reference_provider_has_no_pool():
    run = T.build_run(MODEL, "low", "zai-org")
    assert run["model"] == MODEL and run["reasoning_effort"] == "low" and run["provider"] == "zai-org"
    assert run["route"] == hlb_config.route(MODEL, "zai-org")
    assert run["model_limits"]["context_window"] == 1048576
    assert "providers" not in run
    assert run["benchmark_setups"]["terminal-bench-2.1"] == hlb_config.benchmark_setup("terminal-bench-2.1")


def test_run_on_another_admitted_provider_records_the_pool_like_the_launcher():
    run = T.build_run(MODEL, "low", "novita")
    assert run["provider"] == "zai-org"  # the reference: harnesses keep its route model
    assert set(run["providers"]) == {"zai-org", "novita"}


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        T.build_run(MODEL, "low", "no-such-provider")


def test_prepare_env_gives_the_meter_the_token_and_the_harnesses_a_dummy_key():
    env = {}
    T.prepare_env(env, TOKEN, "alice", "swarm-abc")
    assert env["HF_TOKEN"] == TOKEN and env["HF_INFERENCE_TOKEN"] == TOKEN
    assert env[hlb_config.SANDBOX_KEY_ENV] == env["OPENAI_API_KEY"] != TOKEN
    assert hlb_config.OWNER == "alice"  # sandboxes are inspected and cancelled in the attendee's namespace


def test_job_config_is_one_trial_of_one_harness_on_one_task(tmp_path):
    run = T.build_run(MODEL, "low", "zai-org")
    cfg = T.job_config(run, "pi", "nginx-request-logging", "pi|nginx-request-logging|0", tmp_path, "job1")
    assert cfg["n_attempts"] == 1 and cfg["n_concurrent_trials"] == 1
    assert len(cfg["agents"]) == 1 and "Pi" in cfg["agents"][0]["import_path"]
    assert cfg["datasets"][0]["task_names"] == ["nginx-request-logging"]
    assert cfg["environment"]["import_path"] == "hlb.harbor_ext.environment:HLBSandboxEnvironment"
    labels = cfg["environment"]["kwargs"]["labels"]
    assert labels[T.CELL_LABEL] == hlb_config.label_value("pi|nginx-request-logging|0")
    assert TOKEN not in json.dumps(cfg)


class FakeHarbor:
    """Writes the real smoke trial into the job folder, as Harbor would."""

    def __init__(self):
        self.configs = []

    def __call__(self, cfg):
        self.configs.append(cfg)
        dst = Path(cfg["jobs_dir"]) / cfg["job_name"] / f"{cfg['datasets'][0]['task_names'][0]}__abc123"
        shutil.copytree(FIX, dst)


def test_run_trial_returns_the_trial_folder_and_sets_up_the_environment(tmp_path, monkeypatch):
    harbor = FakeHarbor()
    env = {}
    claim = {"cell_id": "pi|nginx-request-logging|0", "harness": "pi", "task": "nginx-request-logging", "trial": 0, "model": MODEL, "run_config": {}}
    grid = {"reasoning_effort": "low", "provider": "zai-org", "benchmark": "terminal-bench-2.1"}
    d = T.run_trial(claim, grid, TOKEN, "alice", tmp_path, harbor=harbor, environ=env)
    assert (d / "result.json").exists()
    assert env["HF_INFERENCE_TOKEN"] == TOKEN
    assert harbor.configs[0]["datasets"][0]["task_names"] == ["nginx-request-logging"]


def test_run_config_from_the_claim_overrides_the_grid_defaults(tmp_path):
    harbor = FakeHarbor()
    claim = {"cell_id": "c", "harness": "pi", "task": "nginx-request-logging", "trial": 0, "model": MODEL, "run_config": {"provider": "novita", "reasoning_effort": "high"}}
    T.run_trial(claim, {"reasoning_effort": "low", "provider": "zai-org", "benchmark": "terminal-bench-2.1"}, TOKEN, "alice", tmp_path, harbor=harbor, environ={})
    kw = harbor.configs[0]["environment"]["kwargs"]
    assert kw["reasoning_effort"] == "high"
    assert kw["route_model"]  # novita serves the reference's route model through the meter's rewrite


def test_run_trial_without_a_trial_folder_raises(tmp_path):
    claim = {"cell_id": "c", "harness": "pi", "task": "t", "trial": 0, "model": MODEL, "run_config": {}}
    with pytest.raises(T.TrialError):
        T.run_trial(claim, {"reasoning_effort": "low", "provider": "zai-org", "benchmark": "terminal-bench-2.1"}, TOKEN, "alice", tmp_path, harbor=lambda cfg: None, environ={})


class FakeJobsApi:
    def __init__(self, jobs):
        self.jobs, self.asked = jobs, []

    def list_jobs(self, *, labels, namespace):
        self.asked.append((labels, namespace))
        return self.jobs


def test_sandbox_id_is_looked_up_by_the_cell_label_in_the_attendee_namespace():
    class Job:
        id = "687f911eaea852de79c4a50a"

    api = FakeJobsApi([Job()])
    assert T.sandbox_id(api, "pi|t|0", "alice") == "687f911eaea852de79c4a50a"
    assert api.asked == [({T.CELL_LABEL: hlb_config.label_value("pi|t|0")}, "alice")]


def test_sandbox_id_lookup_failure_is_none():
    class Broken:
        def list_jobs(self, **kw):
            raise RuntimeError("503")

    assert T.sandbox_id(Broken(), "c", "alice") is None


def test_parallel_slots_share_one_run_and_sandbox_key_but_never_a_job_folder(tmp_path):
    """Slots of one Job run trials in threads of one process: hlb reads the harnesses' dummy key from os.environ, so
    every trial of the process uses the same run id (and key, as in a leaderboard worker); folders stay apart."""
    seen = []

    def harbor(cfg):
        d = Path(cfg["jobs_dir"]) / cfg["job_name"] / "trial"
        d.mkdir(parents=True)
        seen.append((cfg["job_name"], dict(env)))

    env = {}
    claim = {"cell_id": "pi|t|0", "harness": "pi", "task": "nginx-request-logging", "trial": 0, "model": MODEL, "run_config": {}}
    grid = {"reasoning_effort": "low", "provider": "zai-org"}
    a = T.run_trial(claim, grid, TOKEN, "alice", tmp_path, harbor=harbor, environ=env)
    b = T.run_trial(claim, grid, TOKEN, "alice", tmp_path, harbor=harbor, environ=env)
    assert a != b and seen[0][0] != seen[1][0]
    assert seen[0][1][hlb_config.SANDBOX_KEY_ENV] == seen[1][1][hlb_config.SANDBOX_KEY_ENV]
