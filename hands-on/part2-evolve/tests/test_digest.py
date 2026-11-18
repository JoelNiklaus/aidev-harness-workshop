"""What the proposer reads: failed and passed trajectories rendered as short markdown, failures first."""

from aidev_evolve import digest


def traj(end="returned", n=3, out="x" * 5000):
    steps = []
    for i in range(n):
        steps.append({"kind": "llm", "t": i * 2.0, "duration_s": 1.0, "n_messages": 2 + 2 * i, "dropped_or_changed_from": None,
                      "new_messages": ([{"role": "tool", "tool_call_id": f"c{i-1}", "content": "exit code: 0\n" + "y" * 1000}] if i else []),
                      "reply": {"role": "assistant", "content": f"thinking {i}", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": '{"command": "make step%d"}' % i}}]},
                      "tools": ["bash"], "params": {}, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0}})
        steps.append({"kind": "shell", "t": i * 2.0 + 1, "command": f"make step{i}", "timeout": None, "exit_code": 2, "timed_out": False,
                      "duration_s": 0.5, "output": out, "output_chars": len(out)})
    return {"task": "Build the thing in /app and run its tests.", "end": end, "error": None, "steps": steps,
            "usage": {"input": 100, "cached": 0, "output": 10, "calls": n}, "cost_usd": 0.001, "agent_s": 42.0, "budget_s": 150}


def rec(task, outcome, t=None):
    return {"candidate": "c000", "task": task, "trial_name": f"{task}__x", "outcome": outcome, "reward": 1.0 if outcome == "pass" else 0.0,
            "wall_s": 60.0, "cost_usd": 0.003, "trajectory": t or traj()}


def test_one_trajectory_reads_like_a_transcript():
    md = digest.render(rec("db-wal-recovery", "fail"))
    assert "db-wal-recovery" in md and "FAIL" in md and "ended: returned" in md
    assert "Build the thing" in md
    assert "$ make step1" in md and "exit 2" in md
    assert "5,000 chars" in md  # the size of what the command printed
    assert "model saw 1,013 chars" in md  # what the harness passed on


def _first_messages(t, user_text):
    t["steps"][0]["new_messages"] = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": user_text}]
    return t


def test_the_first_step_says_what_the_model_received_of_the_task():
    # A harness can cut or rewrite the task before the model sees it: the transcript must make that visible.
    t = traj()
    task = t["task"] + " Then write the report to /app/out.txt in JSON."
    t["task"] = task
    md = digest.render(rec("x", "fail", _first_messages(t, task[:30])))
    assert f"model received 30 of the task's {len(task)} chars" in md
    t2 = traj()
    md2 = digest.render(rec("x", "fail", _first_messages(t2, t2["task"])))
    assert "of the task's" not in md2  # the whole task arrived: nothing to report


def test_long_trajectories_are_cut_to_a_budget_keeping_the_end():
    md = digest.render(rec("x", "fail", traj(n=60)), max_chars=12_000)
    assert len(md) <= 12_500
    assert "$ make step59" in md and "steps omitted" in md


def test_write_puts_failures_first_in_the_index(tmp_path):
    recs = [rec("a", "pass"), rec("b", "fail"), rec("c", "infra")]
    digest.write(recs, tmp_path)
    index = (tmp_path / "INDEX.md").read_text()
    assert index.index("| b |") < index.index("| a |")
    assert (tmp_path / "b__1.md").exists() and (tmp_path / "a__1.md").exists()
    assert not (tmp_path / "c__1.md").exists()  # infra trials say nothing about the harness
