"""Harbor agent that runs one candidate agent.py on one task.

The candidate's source arrives as an agent kwarg (the Job's spec carries it), so nothing is installed in
the task container: the agent loop runs in the HF Job and reaches the sandbox through Harbor's
environment (`exec`, which HF Sandbox runs as one process group and kills at its timeout). Each agent
gets its own thread, so 40 concurrent trials do not wait on asyncio's small default thread pool, which
the sandbox exec itself needs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent

from aidev_evolve import runtime
from aidev_evolve.providers import REFERENCE
from aidev_evolve.runtime import ShellResult

TRAJECTORY = "evolve-trajectory.json"
EXEC_SLACK_S = 15  # wait this long past a command's timeout for the sandbox to report it, then give up on it (stays well inside Harbor's own agent timeout)


class EvolveAgent(BaseAgent):
    def __init__(self, logs_dir: Path, model_name: str | None = None, *, source: str, candidate: str = "candidate", budget_s: float = 150,
                 providers: list[str] | None = None, llm: Any = None, **kwargs: Any):
        self._source, self._candidate, self._budget_s, self._llm = source, candidate, float(budget_s), llm
        self._providers = list(providers or [REFERENCE])
        super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "aidev-evolve"

    def version(self) -> str:
        return hashlib.sha256(self._source.encode()).hexdigest()[:12]

    async def setup(self, environment: Any) -> None:
        return None

    def _bridge(self, environment: Any, loop: asyncio.AbstractEventLoop):
        def shell(command: str, timeout: float) -> ShellResult:
            seconds = max(1, math.ceil(timeout))
            start = time.monotonic()
            fut = asyncio.run_coroutine_threadsafe(environment.exec(command, timeout_sec=seconds), loop)
            try:
                r = fut.result(timeout=seconds + EXEC_SLACK_S)
            except TimeoutError:
                fut.cancel()
                return ShellResult(-1, "", timed_out=True)
            except Exception as e:  # the sandbox, not the agent, failed
                if "characters of output" in str(e):  # the command flooded stdout (64 MB): the agent's doing, it can recover
                    return ShellResult(-1, "[the command printed too much output (over 64 MB) and was stopped; redirect it to a file]")
                raise runtime.InfraError(f"sandbox exec failed: {type(e).__name__}: {e}") from None
            took = time.monotonic() - start
            output = (r.stdout or "") + (r.stderr or "")
            return ShellResult(r.return_code, output, timed_out=r.return_code != 0 and took >= seconds - 0.05)

        return shell

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        loop = asyncio.get_running_loop()
        llm = self._llm
        if llm is None:
            from aidev_evolve import llm as llm_mod
            from aidev_evolve.providers import shared_pool

            llm = llm_mod.RouterLLM(os.environ["HF_TOKEN"], pool=shared_pool(self._providers))
        done: asyncio.Future[dict[str, Any]] = loop.create_future()

        def work() -> None:
            try:
                traj = runtime.run_agent(self._source, instruction, shell=self._bridge(environment, loop), llm=llm, budget_s=self._budget_s)
            except BaseException as e:  # noqa: BLE001 - never leave the trial waiting on a dead thread
                loop.call_soon_threadsafe(done.set_exception, RuntimeError(f"agent thread died: {e!r}"))
                return
            loop.call_soon_threadsafe(done.set_result, traj)

        threading.Thread(target=work, name=f"agent-{self._candidate}", daemon=True).start()
        traj = await done
        traj["candidate"] = self._candidate
        traj["provider_calls"] = dict(getattr(llm, "provider_calls", {}))
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.logs_dir) / TRAJECTORY).write_text(json.dumps(traj))
        u = traj["usage"]
        context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens = u["input"], u["cached"], u["output"]
        context.cost_usd = traj["cost_usd"]
        context.metadata = {"end": traj["end"], "candidate": self._candidate, "agent_s": traj["agent_s"]}
