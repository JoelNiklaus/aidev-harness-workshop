"""The runtime agent.py runs in: a pinned model, a shell in the task sandbox, a hard time budget, and a
trajectory recorded outside the agent's reach (the proposer cannot change what the grader records)."""

import pytest

from aidev_evolve import runtime
from aidev_evolve.runtime import ShellResult


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def script_llm(replies, clock=None, cost_s=1.0):
    """An llm backend that plays back assistant messages and advances the fake clock per call."""
    calls = []

    def llm(messages, tools, timeout, **params):
        calls.append({"messages": [dict(m) for m in messages], "tools": tools, "timeout": timeout, "params": params})
        if clock is not None:
            clock.t += cost_s
        if not replies:
            raise AssertionError("unplanned llm call")
        return replies.pop(0), {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 40}

    llm.calls = calls
    return llm


def tool_call(cmd, id="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": id, "type": "function", "function": {"name": "bash", "arguments": '{"command": "%s"}' % cmd}}]}


AGENT = '''
def run(task, env):
    messages = [{"role": "user", "content": task}]
    while True:
        reply = env.llm(messages)
        messages.append(reply)
        if not reply.get("tool_calls"):
            return
        r = env.shell("echo hi")
        messages.append({"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": r.output})
'''


def test_records_llm_and_shell_steps_with_usage_and_cost():
    shells = []

    def shell(cmd, timeout):
        shells.append((cmd, timeout))
        return ShellResult(0, "hi\n")

    llm = script_llm([tool_call("echo hi"), {"role": "assistant", "content": "done"}])
    traj = runtime.run_agent(AGENT, "say hi", shell=shell, llm=llm, budget_s=100, price={"input": 0.15, "cached_input": 0.03, "output": 0.5})
    kinds = [s["kind"] for s in traj["steps"]]
    assert kinds == ["llm", "shell", "llm"]
    assert traj["end"] == "returned" and traj["error"] is None
    assert traj["usage"] == {"input": 200, "cached": 80, "output": 20, "calls": 2}
    assert traj["cost_usd"] == pytest.approx((60 * 0.15 + 40 * 0.03 + 10 * 0.5) * 2 / 1e6)
    assert traj["steps"][1]["command"] == "echo hi" and traj["steps"][1]["exit_code"] == 0
    # the second call's record holds only what the agent added since the first call
    assert [m["role"] for m in traj["steps"][2]["new_messages"]] == ["assistant", "tool"]


def test_shell_timeout_is_capped_by_the_time_left():
    clock = FakeClock()
    seen = []

    def shell(cmd, timeout):
        seen.append(timeout)
        return ShellResult(0, "")

    agent = "def run(task, env):\n    env.shell('a')\n    env.shell('b', timeout=5)\n"
    runtime.run_agent(agent, "t", shell=shell, llm=script_llm([]), budget_s=60, clock=clock)
    assert seen == [60, 5]


def test_a_command_that_runs_into_the_deadline_ends_the_run_as_timeout():
    clock = FakeClock()

    def shell(cmd, timeout):
        clock.t += timeout
        return ShellResult(-1, "server listening on :8080\n", timed_out=True)

    agent = "def run(task, env):\n    env.shell('python server.py')\n    env.shell('never')\n"
    traj = runtime.run_agent(agent, "t", shell=shell, llm=script_llm([]), budget_s=30, clock=clock)
    assert traj["end"] == "timeout"
    assert [s.get("command") for s in traj["steps"] if s["kind"] == "shell"] == ["python server.py"]
    assert traj["steps"][0]["timed_out"] is True


def test_an_explicit_short_timeout_returns_to_the_agent():
    clock = FakeClock()

    def shell(cmd, timeout):
        clock.t += timeout
        return ShellResult(-1, "partial", timed_out=True)

    agent = "def run(task, env):\n    r = env.shell('sleep 99', timeout=3)\n    env.log('timed_out=%s' % r.timed_out)\n"
    traj = runtime.run_agent(agent, "t", shell=shell, llm=script_llm([]), budget_s=30, clock=clock)
    assert traj["end"] == "returned"
    assert traj["steps"][-1] == {"kind": "note", "t": 3.0, "text": "timed_out=True"}


def test_llm_after_the_deadline_raises_and_the_run_ends_as_timeout():
    clock = FakeClock()
    llm = script_llm([tool_call("x"), tool_call("y")], clock=clock, cost_s=40)
    agent = "def run(task, env):\n    while True:\n        env.llm([{'role': 'user', 'content': task}])\n"
    traj = runtime.run_agent(agent, "t", shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=50, clock=clock)
    assert traj["end"] == "timeout"
    assert len(llm.calls) == 2
    assert llm.calls[1]["timeout"] == 10  # the call gets only the time that is left


def test_the_model_and_effort_are_not_the_agents_to_choose():
    llm = script_llm([{"role": "assistant", "content": "ok"}])
    agent = "def run(task, env):\n    env.llm([{'role': 'user', 'content': task}], model='gpt-5')\n"
    traj = runtime.run_agent(agent, "t", shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=50)
    assert traj["end"] == "error" and "model" in traj["error"]
    assert llm.calls == []


def test_sampling_parameters_pass_through():
    llm = script_llm([{"role": "assistant", "content": "ok"}])
    agent = "def run(task, env):\n    env.llm([{'role': 'user', 'content': task}], tools=[], temperature=0.2, max_tokens=500)\n"
    runtime.run_agent(agent, "t", shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=50)
    assert llm.calls[0]["params"] == {"temperature": 0.2, "max_tokens": 500}


def test_an_agent_crash_is_recorded_with_its_traceback_not_raised():
    agent = "def run(task, env):\n    raise KeyError('arguments')\n"
    traj = runtime.run_agent(agent, "t", shell=lambda c, t: ShellResult(0, ""), llm=script_llm([]), budget_s=50)
    assert traj["end"] == "error"
    assert "KeyError" in traj["error"] and "arguments" in traj["error"]


def test_an_account_problem_ends_the_trial_as_infra_even_through_except_exception():
    def llm(messages, tools, timeout, **params):
        raise runtime.InfraError("router HTTP 402: no credits")

    agent = "def run(task, env):\n    try:\n        env.llm([])\n    except Exception:\n        pass\n    env.shell('x')\n"
    traj = runtime.run_agent(agent, "t", shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=50)
    assert traj["end"] == "infra" and "402" in traj["error"]
    assert not [s for s in traj["steps"] if s["kind"] == "shell"]


def test_source_without_run_function_is_an_error():
    traj = runtime.run_agent("x = 1\n", "t", shell=lambda c, t: ShellResult(0, ""), llm=script_llm([]), budget_s=50)
    assert traj["end"] == "error" and "run(task, env)" in traj["error"]


def test_recorded_outputs_are_capped_head_and_tail():
    out = "START" + "x" * 50_000 + "END"
    agent = "def run(task, env):\n    env.shell('cat big')\n"
    traj = runtime.run_agent(agent, "t", shell=lambda c, t: ShellResult(0, out), llm=script_llm([]), budget_s=50)
    rec = traj["steps"][0]["output"]
    assert rec.startswith("START") and rec.endswith("END") and len(rec) < 9_000
    assert traj["steps"][0]["output_chars"] == len(out)
