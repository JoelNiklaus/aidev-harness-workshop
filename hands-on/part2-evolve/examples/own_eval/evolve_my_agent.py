"""The evolve loop on your own agent and your own tasks, without the workshop's benchmark (APPLY-TO-YOUR-AGENT.md).

    uv run python examples/own_eval/evolve_my_agent.py --proposer claude        # from hands-on/part2-evolve; needs `uvx hf auth login`

Four things are yours: agent.py (the harness the proposer edits), tasks.json (each task with a pass/fail check),
splits.json (search, validation, held-out) and the two functions below that run one trial (`llm`, `evaluate`).
The loop itself is the workshop's (`aidev_evolve.loop.Evolver`): seed calibration, propose, audit, evaluate, keep or
drop with a noise margin, and one held-out check at the end.

Each trial runs the model's shell commands on THIS machine, in a fresh temporary folder. That is fine for the toy
tasks here; for real tasks run the trial in a container or a remote sandbox.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aidev_evolve import launch, loop, proposer, providers, runtime, strategy
from aidev_evolve.llm import RouterLLM

HERE = Path(__file__).parent
TASKS = json.loads((HERE / "tasks.json").read_text())["tasks"]

# What the proposer is told about your setting (the strategy's proposer_prompt.md and the fixed rules follow it).
HEADER = """You are improving a small agent harness: the Python file `agent.py` in this folder.
A fixed model runs inside this harness on small shell tasks; a check script decides pass or fail. Each task run gets
{budget_s:.0f} seconds; the model and the time budget are fixed. Only the harness can change.

This is iteration {iteration}.
"""


def shell_in(folder: Path) -> runtime.ShellBackend:
    """`env.shell` for one trial: bash in the trial's folder, the whole process group killed at the timeout."""
    def shell(command: str, timeout: float) -> runtime.ShellResult:
        p = subprocess.Popen(["bash", "-c", command], cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            return runtime.ShellResult(p.returncode, out)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return runtime.ShellResult(-1, out, timed_out=True)
    return shell


def trial(candidate: str, source: str, name: str, k: int, budget_s: float, llm: Any) -> dict[str, Any]:
    """One task run: set up a folder, run the agent, run the task's check. Returns the record the loop reads."""
    task = TASKS[name]
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="own-eval-") as tmp:
        subprocess.run(["bash", "-c", task["setup"]], cwd=tmp, check=True, capture_output=True)
        traj = runtime.run_agent(source, task["task"], shell=shell_in(Path(tmp)), llm=llm, budget_s=budget_s)
        passed = subprocess.run(["bash", "-c", task["check"]], cwd=tmp, capture_output=True).returncode == 0
    outcome = "infra" if traj["end"] == "infra" else ("pass" if passed else "fail")
    return {"candidate": candidate, "task": name, "trial_name": f"{name}__{k}", "outcome": outcome, "reward": float(passed),
            "wall_s": round(time.monotonic() - start, 1), "cost_usd": traj["cost_usd"], "trajectory": traj}


def make_evaluate(make_llm: Any, workers: int = 6) -> Any:
    """The loop's evaluator: `evaluate(spec, on_record)` runs every (candidate, task, trial) of the spec."""
    def evaluate(spec: dict[str, Any], on_record: Any) -> launch.EvalResult:
        start = time.monotonic()
        jobs = [(r["candidate"], spec["candidates"][r["candidate"]], t, k) for r in spec["runs"] for t in r["tasks"] for k in range(r["trials"])]
        with ThreadPoolExecutor(workers) as pool:
            records = list(pool.map(lambda j: trial(*j, spec["budget_s"], make_llm()), jobs))
        for rec in records:
            on_record(rec)
        return launch.EvalResult(records, "local", None, time.monotonic() - start, 0.0)
    return evaluate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposer", choices=["claude", "codex", "manual"], default="claude")
    ap.add_argument("--strategy", default="code-audit", help="a preset or a folder from `aidev-evolve strategy-init`")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--run-dir", default=f"runs/own-eval-{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, get_token

    token = get_token()
    user = HfApi(token=token).whoami()["name"]
    # The frozen model: GLM-5.3-Flash on the HF router. Any function `llm(messages, tools, *, timeout, **params)` that
    # returns (assistant message, {"prompt_tokens", "completion_tokens", "cached_tokens"}) works here.
    pool = providers.Pool(providers.for_user(user))
    evo = loop.Evolver(Path(a.run_dir), splits=json.loads((HERE / "splits.json").read_text()), seed_source=(HERE / "agent.py").read_text(),
                       evaluate=make_evaluate(lambda: RouterLLM(token, pool=pool)),
                       propose=lambda work, prompt: proposer.propose(a.proposer, work, prompt),
                       strategy=strategy.load(a.strategy), proposer_name=a.proposer, proposer_header=HEADER,
                       write=lambda s: print(s, end="", flush=True))
    evo.run(iterations=a.iterations, heldout_trials=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
