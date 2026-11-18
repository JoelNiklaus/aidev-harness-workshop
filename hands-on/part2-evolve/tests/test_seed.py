"""The starting harness: small, working, and deliberately naive (it must leave room to improve)."""

from importlib import resources

from aidev_evolve import runtime
from aidev_evolve.runtime import ShellResult

from test_runtime import script_llm, tool_call

SEED = resources.files("aidev_evolve.seed").joinpath("agent.py").read_text()


def test_seed_is_short_enough_to_read_on_a_slide_deck():
    lines = [l for l in SEED.splitlines() if l.strip()]
    assert 30 <= len(lines) <= 150


def test_seed_runs_bash_calls_and_returns_the_output_to_the_model():
    seen = []

    def shell(cmd, timeout):
        seen.append(cmd)
        return ShellResult(0, "file.txt\n")

    llm = script_llm([tool_call("ls"), {"role": "assistant", "content": "Done."}])
    traj = runtime.run_agent(SEED, "List the files.", shell=shell, llm=llm, budget_s=100)
    assert len(seen) == 1 and seen[0].startswith("{ ls\n") and traj["end"] == "returned"  # Phase 2 seed wraps the command (stderr hidden)
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "file.txt" in tool_msgs[0]["content"] and tool_msgs[0]["tool_call_id"] == "c1"
    assert llm.calls[0]["tools"][0]["function"]["name"] == "bash"


def test_seed_stops_at_its_step_limit():
    n = 200
    llm = script_llm([tool_call("true", id=f"c{i}") for i in range(n)])
    traj = runtime.run_agent(SEED, "loop", shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=10_000)
    assert traj["end"] == "returned" and 10 <= len(llm.calls) < n


def test_seed_shows_the_model_only_the_head_of_long_outputs():
    long = "line\n" * 5000 + "ERROR: the real problem is here\n"
    llm = script_llm([tool_call("make"), {"role": "assistant", "content": "ok"}])
    runtime.run_agent(SEED, "build", shell=lambda c, t: ShellResult(2, long), llm=llm, budget_s=100)
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "the real problem" not in tool_msg["content"]  # a weakness the proposer can find in the traces


def test_seed_hides_stderr_from_the_model():
    """Phase 2 seed: only stdout reaches the model (error messages are lost; a weakness the traces show)."""
    seen = []

    def shell(cmd, timeout):
        seen.append(cmd)
        return ShellResult(127, "")

    llm = script_llm([tool_call("pdftotext x.pdf -"), {"role": "assistant", "content": "Done."}])
    runtime.run_agent(SEED, "Read the PDF.", shell=shell, llm=llm, budget_s=100)
    assert "pdftotext x.pdf -" in seen[0] and "2>/dev/null" in seen[0]


def test_seed_step_limit_is_ten():
    n = 50
    llm = script_llm([tool_call("true", id=f"c{i}") for i in range(n)])
    runtime.run_agent(SEED, "loop", shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=10_000)
    assert len(llm.calls) == 10


def test_bundled_splits_carry_their_budget_and_selection_record():
    import json

    splits = json.loads(resources.files("aidev_evolve.data").joinpath("splits.json").read_text())
    assert splits["budget_s"] == 180
    every = splits["search"] + splits["validation"] + splits["heldout"]
    assert len(every) == len(set(every)) and "headless-terminal" not in every
    # Phase 4 (splits E): each seed weakness bites validation tasks (so its fix can be kept) and held-out tasks (so
    # finding it shows in the strategy ranking); "bites" = fixing it adds >= 2 of 4 passes in the paired sweep.
    cal = splits["calibration"]
    for weakness in ("step limit", "stray-process kill", "task cut"):
        assert sum(weakness in cal[t]["bites"] for t in splits["validation"]) >= 2, weakness
        assert sum(weakness in cal[t]["bites"] for t in splits["heldout"]) >= 2, weakness


def test_seed_kills_every_command_after_ten_seconds():
    """Phase 3 seed C: the second weakness (long builds and servers get killed); its fix is a different mechanism
    from the step limit (sweep 18 Nov: +13 points, step fix +22)."""
    timeouts = []

    def shell(cmd, timeout=None):
        timeouts.append(timeout)
        return ShellResult(0, "ok\n")

    llm = script_llm([tool_call("c1", "make -j4"), {"role": "assistant", "content": "done"}])
    runtime.run_agent(SEED, "build it", shell=shell, llm=llm, budget_s=180)
    assert timeouts and all(t == 10 for t in timeouts)


def test_seed_kills_whatever_a_command_left_running():
    """Phase 4 seed E: a server the model started (and checked) is gone after its command; no transcript shows it
    failing, only the cleanup appended to every command (sweep 19 Nov: -8 pts with the other weaknesses fixed)."""
    seen = []

    def shell(cmd, timeout=None):
        seen.append(cmd)
        return ShellResult(0, "ok\n")

    llm = script_llm([tool_call("nohup python -m http.server 8000 &"), {"role": "assistant", "content": "done"}])
    runtime.run_agent(SEED, "serve it", shell=shell, llm=llm, budget_s=180)
    assert "nohup python -m http.server 8000 &" in seen[0] and "kill -9" in seen[0].split("http.server 8000 &", 1)[1]


def test_seed_cuts_the_task_text_at_1000_chars():
    """Phase 4 seed E: requirements at the end of a long task never reach the model (-19 pts with the rest fixed)."""
    task = "Do the work. " * 100 + "Write the result to /app/out.json."
    llm = script_llm([{"role": "assistant", "content": "done"}])
    runtime.run_agent(SEED, task, shell=lambda c, t: ShellResult(0, ""), llm=llm, budget_s=100)
    user = [m for m in llm.calls[0]["messages"] if m["role"] == "user"][0]["content"]
    assert len(user) == 1000 and "/app/out.json" not in user


def test_bundled_held_out_set_is_every_pool_task_the_loop_never_sees():
    import json

    splits = json.loads(resources.files("aidev_evolve.data").joinpath("splits.json").read_text())
    assert len(splits["heldout"]) >= 20  # enough tasks to rank strategies (devtools/strategy_noise.py)
    assert not set(splits["heldout"]) & set(splits["search"] + splits["validation"])
