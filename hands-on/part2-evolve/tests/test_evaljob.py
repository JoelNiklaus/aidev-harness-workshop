"""What runs inside the HF Job: Harbor trials of each candidate, results streamed back through the log."""

import json
from pathlib import Path

from aidev_evolve import evaljob, transport

SEED = "def run(task, env):\n    pass\n"


def spec(**over):
    s = {"candidates": {"c000": SEED, "c001": SEED.replace("pass", "return")},
         "runs": [{"candidate": "c001", "tasks": ["fix-git", "regex-log"], "trials": 1},
                  {"candidate": "c000", "tasks": ["regex-log"], "trials": 2}],
         "budget_s": 150}
    s.update(over)
    return s


def write_trial(jobs_dir, job_name, task, i, *, reward=1.0, end="returned", exception=None, traj=True, cost=0.002):
    d = Path(jobs_dir) / job_name / f"{task}__{i}"
    (d / "agent").mkdir(parents=True)
    result = {"task_name": f"terminal-bench/{task}", "trial_name": d.name,
              "started_at": "2026-11-17T10:00:00Z", "finished_at": "2026-11-17T10:02:00Z",
              "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
              "exception_info": exception, "agent_execution": {"started_at": "2026-11-17T10:00:10Z", "finished_at": "2026-11-17T10:01:30Z"}}
    (d / "result.json").write_text(json.dumps(result))
    if traj:
        (d / "agent" / "evolve-trajectory.json").write_text(json.dumps(
            {"candidate": "x", "end": end, "error": None, "steps": [], "usage": {"input": 10, "cached": 0, "output": 1, "calls": 1}, "cost_usd": cost, "agent_s": 80}))
    return d


def test_one_harbor_job_per_candidate_with_the_source_as_agent_kwarg(tmp_path):
    cfgs = evaljob.harbor_configs(spec(), tmp_path, run_id="r1")
    assert [c["job_name"] for c in cfgs] == ["r1-0-c001", "r1-1-c000"]
    agent = cfgs[0]["agents"][0]
    assert agent["import_path"] == "aidev_evolve.harbor_agent:EvolveAgent"
    assert agent["kwargs"]["source"] == spec()["candidates"]["c001"] and agent["kwargs"]["budget_s"] == 150
    assert agent["override_timeout_sec"] > 150  # Harbor's hard stop stays behind the runtime's own budget
    assert cfgs[0]["datasets"][0]["task_names"] == ["fix-git", "regex-log"]
    assert cfgs[1]["n_attempts"] == 2 and cfgs[1]["n_concurrent_trials"] == 2
    assert cfgs[0]["environment"]["import_path"] == "hlb.harbor_ext.environment:HLBSandboxEnvironment"
    assert cfgs[0]["verifier"]["override_timeout_sec"] == evaljob.VERIFIER_CAP_S  # one hung test run cannot hold the iteration for 15 min


def test_a_verifier_timeout_is_not_scored(tmp_path):
    t = write_trial(tmp_path, "j", "a", 0, reward=None, exception={"exception_type": "VerifierTimeoutError", "exception_message": "240 s"})
    assert evaljob.record(t, "c")["outcome"] == "infra"


def test_record_from_a_finished_trial(tmp_path):
    d = write_trial(tmp_path, "r1-c001", "fix-git", 0, reward=1.0)
    rec = evaljob.record(d, "c001")
    assert rec["candidate"] == "c001" and rec["task"] == "fix-git"
    assert rec["outcome"] == "pass" and rec["reward"] == 1.0 and rec["wall_s"] == 120.0
    assert rec["cost_usd"] > 0.002  # inference plus the sandbox estimate
    assert rec["trajectory"]["end"] == "returned"


def test_agent_timeout_with_a_verdict_is_scored_but_a_sandbox_start_failure_is_infra(tmp_path):
    t = write_trial(tmp_path, "j", "a", 0, reward=0.0, end="timeout", exception={"exception_type": "AgentTimeoutError", "exception_message": "x"})
    assert evaljob.record(t, "c")["outcome"] == "fail"
    s = write_trial(tmp_path, "j", "b", 0, reward=None, traj=False, exception={"exception_type": "SandboxError", "exception_message": "start"})
    assert evaljob.record(s, "c")["outcome"] == "infra"
    i = write_trial(tmp_path, "j", "c", 0, reward=0.0, end="infra")
    assert evaljob.record(i, "c")["outcome"] == "infra"


def test_main_streams_each_record_once_and_ends_with_the_done_line(tmp_path):
    lines = []

    def fake_harbor(cfgs):
        for cfg in cfgs:
            cand = cfg["agents"][0]["kwargs"]["candidate"]
            for t in cfg["datasets"][0]["task_names"]:
                for i in range(cfg["n_attempts"]):
                    write_trial(cfg["jobs_dir"], cfg["job_name"], t, f"{cand}{i}")

    n = evaljob.main(spec(), workdir=tmp_path, harbor=fake_harbor, emit=lines.append, prepare=lambda run: None, poll_s=0.01)
    dec = transport.Decoder()
    recs = [r for line in lines for r in dec.feed(line)]
    assert n == 4 and len(recs) == 4
    assert sorted((r["candidate"], r["task"]) for r in recs) == [("c000", "regex-log"), ("c000", "regex-log"), ("c001", "fix-git"), ("c001", "regex-log")]
    assert transport.is_done(lines[-1]) and transport.done_count(lines[-1]) == 4


def test_two_runs_of_the_same_candidate_get_separate_harbor_jobs(tmp_path):
    s = spec(runs=[{"candidate": "c000", "tasks": ["fix-git"], "trials": 1}, {"candidate": "c000", "tasks": ["regex-log"], "trials": 2}])
    names = [c["job_name"] for c in evaljob.harbor_configs(s, tmp_path, run_id="r1")]
    assert len(set(names)) == 2


def test_a_harbor_job_that_raises_is_reported_in_the_log(tmp_path):
    lines = []

    def broken(cfgs):
        raise FileExistsError("job dir exists")

    evaljob.main(spec(), workdir=tmp_path, harbor=broken, emit=lines.append, prepare=lambda run: None, poll_s=0.01)
    assert any("FileExistsError" in l for l in lines) and transport.is_done(lines[-1])


def test_spec_roundtrip_through_the_environment_variable():
    s = spec()
    assert evaljob.decode_spec(evaljob.encode_spec(s)) == s
