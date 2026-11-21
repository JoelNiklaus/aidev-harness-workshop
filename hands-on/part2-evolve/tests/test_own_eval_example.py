"""examples/own_eval: the loop on your own agent and tasks (APPLY-TO-YOUR-AGENT.md). The model and the proposer are
scripted here; the trial runner, the task checks and the loop are the example's and the package's own."""

import importlib.util
import json
from pathlib import Path

import pytest

from aidev_evolve import audit, loop, strategy
from aidev_evolve.proposer import ProposerResult

EX = Path(__file__).parents[1] / "examples" / "own_eval"
spec = importlib.util.spec_from_file_location("evolve_my_agent", EX / "evolve_my_agent.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

SPLITS = json.loads((EX / "splits.json").read_text())
SEED = (EX / "agent.py").read_text()


def scripted_llm(commands):
    """A model that asks for each command in turn, then answers in text."""
    def make():
        left = list(commands)

        def llm(messages, tools, *, timeout, **params):
            if left:
                call = {"id": f"c{len(left)}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": left.pop(0)})}}
                return {"role": "assistant", "content": None, "tool_calls": [call]}, {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0}
            return {"role": "assistant", "content": "Done."}, {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0}
        return llm
    return make


def test_every_split_task_exists_and_no_task_is_in_two_splits():
    names = [t for k in ("search", "validation", "heldout") for t in SPLITS[k]]
    assert len(names) == len(set(names)) and set(names) <= set(ex.TASKS)
    assert all({"task", "setup", "check"} <= set(t) for t in ex.TASKS.values())


def test_the_seed_agent_passes_the_loops_own_checks_before_any_run():
    assert audit.problems(SEED, list(ex.TASKS)) == [] and loop.preflight(SEED) is None


def test_a_trial_passes_when_the_agent_solved_the_task_and_fails_otherwise():
    ok = ex.trial("c000", SEED, "count-lines-1", 0, 30, scripted_llm(["wc -l < data.txt > answer.txt"])())
    assert ok["outcome"] == "pass" and ok["trajectory"]["steps"] and ok["task"] == "count-lines-1"
    bad = ex.trial("c000", SEED, "count-lines-1", 0, 30, scripted_llm(["echo 3 > answer.txt"])())
    assert bad["outcome"] == "fail"


def test_the_seeds_one_second_command_limit_fails_the_slow_tool_and_a_longer_limit_passes_it():
    """The planted weakness is real: the same model behaviour fails under the seed and passes once the limit is raised."""
    cmd = ["./tool.sh | sed 's/code: //' > answer.txt"]
    assert ex.trial("c000", SEED, "slow-tool-1", 0, 30, scripted_llm(cmd)())["outcome"] == "fail"
    fixed = SEED.replace("COMMAND_TIMEOUT_S = 1", "COMMAND_TIMEOUT_S = 20")
    assert ex.trial("c001", fixed, "slow-tool-1", 0, 30, scripted_llm(cmd)())["outcome"] == "pass"


def test_the_loop_runs_end_to_end_on_the_example_and_keeps_the_fix(tmp_path):
    """Seed calibration, one proposal, keep, held-out: with a model that always runs the slow tool the same way."""
    def model():
        def llm(messages, tools, *, timeout, **params):
            task = messages[1]["content"]
            done = any(m.get("role") == "tool" for m in messages)
            if done:
                return {"role": "assistant", "content": "Done."}, {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}
            cmd = ("./tool.sh | sed 's/code: //' > answer.txt" if "tool.sh" in task else
                   "./report.sh | tail -n 1 | sed 's/TOTAL=//' > answer.txt" if "report.sh" in task else "wc -l < data.txt > answer.txt")
            call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
            return {"role": "assistant", "content": None, "tool_calls": [call]}, {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}
        return llm

    def propose(work, prompt):
        assert "small shell tasks" in prompt and "Terminal-Bench" not in prompt  # the example's own header reached the proposer
        p = Path(work) / "agent.py"
        p.write_text(p.read_text().replace("COMMAND_TIMEOUT_S = 1", "COMMAND_TIMEOUT_S = 20"))
        (Path(work) / "PATCH.md").write_text("# Longer command limit\n\nHypothesis: x\nChange: y\nExpected: z\n")
        return ProposerResult(True, 1.0)

    out = []
    evo = loop.Evolver(tmp_path / "run", splits=SPLITS, seed_source=SEED, evaluate=ex.make_evaluate(model), propose=propose,
                       strategy=strategy.load("code-audit"), proposer_header=ex.HEADER, write=out.append)
    state = evo.run(iterations=1, heldout_trials=1)
    assert state["incumbent"] == "c001" and state["history"][0]["verdict"] == "kept"
    assert state["heldout"]["seed"] == pytest.approx(0.25) and state["heldout"]["best"] == 1.0  # the three slow tools fail under the seed
    assert (tmp_path / "run" / "final_agent.py").read_text().count("COMMAND_TIMEOUT_S = 20") == 1
