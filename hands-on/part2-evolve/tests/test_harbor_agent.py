"""The Harbor agent that runs a candidate agent.py: it lives in the HF Job and drives the task sandbox
through Harbor's environment (no install inside the task container)."""

import asyncio
import json
import time

from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from aidev_evolve.harbor_agent import EvolveAgent
from aidev_evolve.runtime import ShellResult

AGENT = '''
def run(task, env):
    r = env.shell("echo " + task)
    env.llm([{"role": "user", "content": r.output}])
    r2 = env.shell("sleep 5", timeout=0.2)
    env.log("second timed_out=%s exit=%s" % (r2.timed_out, r2.exit_code))
'''


class FakeEnvironment:
    def __init__(self):
        self.calls = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append((command, timeout_sec))
        if command.startswith("sleep"):
            await asyncio.sleep(min(float(command.split()[1]), timeout_sec or 99))
            return ExecResult(stdout="", stderr="", return_code=-1)
        return ExecResult(stdout=command[5:] + "\n", stderr="warn\n", return_code=0)


def fake_llm(messages, tools, timeout, **params):
    return {"role": "assistant", "content": "ok"}, {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0}


def test_runs_the_candidate_and_writes_its_trajectory_and_usage(tmp_path):
    agent = EvolveAgent(logs_dir=tmp_path, model_name="zai-org/GLM-5.3-Flash", source=AGENT, candidate="c001", budget_s=30, llm=fake_llm)
    env = FakeEnvironment()
    ctx = AgentContext()
    asyncio.run(agent.run("hello", env, ctx))
    traj = json.loads((tmp_path / "evolve-trajectory.json").read_text())
    assert traj["candidate"] == "c001" and traj["end"] == "returned"
    shell = [s for s in traj["steps"] if s["kind"] == "shell"]
    assert shell[0]["output"] == "hello\nwarn\n"  # stdout and stderr together
    assert shell[1]["timed_out"] is True
    assert traj["steps"][-1]["text"].startswith("second timed_out=True")
    assert env.calls[1] == ("sleep 5", 1)  # the sandbox enforces whole seconds, rounded up
    assert ctx.n_input_tokens == 1000 and ctx.n_output_tokens == 100 and ctx.cost_usd > 0


class CountingLLM:
    def __init__(self):
        self.provider_calls = {"novita": 1}

    def __call__(self, messages, tools, timeout, **params):
        return fake_llm(messages, tools, timeout, **params)


def test_the_trajectory_records_which_providers_answered(tmp_path):
    agent = EvolveAgent(logs_dir=tmp_path, model_name="m", source=AGENT, budget_s=30, llm=CountingLLM())
    asyncio.run(agent.run("hello", FakeEnvironment(), AgentContext()))
    assert json.loads((tmp_path / "evolve-trajectory.json").read_text())["provider_calls"] == {"novita": 1}


def test_trials_of_one_job_call_the_router_through_one_shared_provider_pool(tmp_path, monkeypatch):
    from aidev_evolve import llm, providers

    pools = []

    class Router(CountingLLM):
        def __init__(self, token, *, pool):
            super().__init__()
            pools.append(pool)

    monkeypatch.setattr(llm, "RouterLLM", Router)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    for i in range(2):
        (tmp_path / str(i)).mkdir()
        a = EvolveAgent(logs_dir=tmp_path / str(i), model_name="m", source=AGENT, budget_s=30, providers=["novita", "zai-org"])
        asyncio.run(a.run("hello", FakeEnvironment(), AgentContext()))
    assert pools[0] is pools[1] is providers.shared_pool(["novita", "zai-org"])


def test_many_agents_run_at_once_without_starving_the_event_loop(tmp_path):
    source = "def run(task, env):\n    env.shell('sleep 0.3', timeout=5)\n"

    async def many():
        start = time.monotonic()
        agents = [EvolveAgent(logs_dir=tmp_path / str(i), model_name="m", source=source, budget_s=30, llm=fake_llm) for i in range(40)]
        for i in range(40):
            (tmp_path / str(i)).mkdir()
        await asyncio.gather(*(a.run("t", FakeEnvironment(), AgentContext()) for a in agents))
        return time.monotonic() - start

    assert asyncio.run(many()) < 3.0  # 40 agents in parallel, not 6 at a time


def test_a_command_with_huge_output_is_the_agents_problem_not_infra(tmp_path):
    class Flood(FakeEnvironment):
        async def exec(self, command, **kw):
            raise RuntimeError("SandboxError: command produced more than 67108864 characters of output. Pass `capture_output=False`")

    source = "def run(task, env):\n    r = env.shell('cat /dev/urandom | base64')\n    env.log('exit=%s out=%s' % (r.exit_code, r.output[:40]))\n"
    agent = EvolveAgent(logs_dir=tmp_path, model_name="m", source=source, budget_s=30, llm=fake_llm)
    asyncio.run(agent.run("t", Flood(), AgentContext()))
    traj = json.loads((tmp_path / "evolve-trajectory.json").read_text())
    assert traj["end"] == "returned" and "exit=-1" in traj["steps"][-1]["text"] and "too much output" in traj["steps"][-1]["text"]


def test_a_sandbox_failure_is_infra_not_an_agent_crash(tmp_path):
    class Broken(FakeEnvironment):
        async def exec(self, command, **kw):
            raise ConnectionError("sandbox gone")

    source = "def run(task, env):\n    try:\n        env.shell('ls')\n    except Exception:\n        pass\n"
    agent = EvolveAgent(logs_dir=tmp_path, model_name="m", source=source, budget_s=30, llm=fake_llm)
    asyncio.run(agent.run("t", Broken(), AgentContext()))
    traj = json.loads((tmp_path / "evolve-trajectory.json").read_text())
    assert traj["end"] == "infra" and "sandbox gone" in traj["error"]


def test_version_names_the_candidate_source():
    a = EvolveAgent(logs_dir=".", model_name="m", source="def run(task, env):\n    pass\n", candidate="seed", llm=fake_llm)
    b = EvolveAgent(logs_dir=".", model_name="m", source="def run(task, env):\n    return\n", candidate="seed", llm=fake_llm)
    assert a.version() != b.version() and len(a.version()) == 12
