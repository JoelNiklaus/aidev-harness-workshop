"""The runtime agent.py runs in.

agent.py defines `run(task, env)`. `env` is the only door to the world:

- `env.llm(messages, tools=None, *, temperature=None, max_tokens=None, top_p=None)` -> the assistant message
  as a dict (`role`, `content`, optional `tool_calls`). The model, provider and reasoning effort are fixed by
  the experiment, not by the agent.
- `env.shell(command, timeout=None)` -> `ShellResult(exit_code, output, timed_out)`: one `bash -c` in the task
  sandbox (stdout and stderr together). Without a timeout a command may run until the time budget is gone.
- `env.time_left()` -> seconds left of the trial's budget; `env.log(text)` adds a note to the trajectory.

The runtime records every step itself (a change to agent.py cannot change what is recorded) and
ends the run when the budget is gone: `TimeUp` derives from BaseException so `except Exception` in an
agent cannot swallow it.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

GLM_PRICE = {"input": 0.15, "cached_input": 0.03, "output": 0.5}  # USD per 1M tokens (zai-org list price, Part 1 prices.json)
CAP_CHARS = 4_000  # head and tail kept per recorded text


@dataclass
class ShellResult:
    exit_code: int
    output: str
    timed_out: bool = False


class TimeUp(BaseException):
    """The trial's time budget is gone."""


class InfraError(BaseException):
    """Not the harness's fault (token, credits): the trial is not scored."""


ShellBackend = Callable[[str, float], ShellResult]
LLMBackend = Callable[..., tuple[dict[str, Any], dict[str, int]]]


def cap(text: str | None, n: int = CAP_CHARS) -> str | None:
    if text is None or len(text) <= 2 * n:
        return text
    return f"{text[:n]}\n[... {len(text) - 2 * n} chars not recorded ...]\n{text[-n:]}"


def call_cost(usage: dict[str, int], price: dict[str, float]) -> float:
    cached = min(usage.get("cached_tokens", 0), usage.get("prompt_tokens", 0))
    return ((usage.get("prompt_tokens", 0) - cached) * price["input"] + cached * price["cached_input"]
            + usage.get("completion_tokens", 0) * price["output"]) / 1e6


def _capped_message(m: dict[str, Any]) -> dict[str, Any]:
    out = dict(m)
    if isinstance(out.get("content"), str):
        out["content"] = cap(out["content"])
    return out


class Env:
    def __init__(self, shell: ShellBackend, llm: LLMBackend, budget_s: float, clock: Callable[[], float], price: dict[str, float]):
        self._shell, self._llm, self._clock, self._price = shell, llm, clock, price
        self._t0 = clock()
        self._deadline = self._t0 + budget_s
        self._last_messages: list[Any] = []
        self.steps: list[dict[str, Any]] = []
        self.usage = {"input": 0, "cached": 0, "output": 0, "calls": 0}
        self.cost_usd = 0.0

    def _t(self) -> float:
        return round(self._clock() - self._t0, 3)

    def time_left(self) -> float:
        return max(0.0, self._deadline - self._clock())

    def log(self, text: str) -> None:
        self.steps.append({"kind": "note", "t": self._t(), "text": cap(str(text))})

    def shell(self, command: str, timeout: float | None = None) -> ShellResult:
        left = self._deadline - self._clock()
        if left <= 0:
            raise TimeUp
        effective = left if timeout is None else min(timeout, left)
        start = self._clock()
        r = self._shell(command, effective)
        self.steps.append({"kind": "shell", "t": round(start - self._t0, 3), "command": cap(command), "timeout": timeout,
                           "exit_code": r.exit_code, "timed_out": r.timed_out, "duration_s": round(self._clock() - start, 3),
                           "output": cap(r.output), "output_chars": len(r.output or "")})
        if r.timed_out and (timeout is None or timeout >= left):
            raise TimeUp
        return r

    def llm(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *, temperature: float | None = None,
            max_tokens: int | None = None, top_p: float | None = None) -> dict[str, Any]:
        left = self._deadline - self._clock()
        if left <= 0:
            raise TimeUp
        params = {k: v for k, v in (("temperature", temperature), ("max_tokens", max_tokens), ("top_p", top_p)) if v is not None}
        snapshot = json.loads(json.dumps(messages, default=str))
        same = 0
        while same < min(len(snapshot), len(self._last_messages)) and snapshot[same] == self._last_messages[same]:
            same += 1
        start = self._clock()
        try:
            reply, usage = self._llm(messages, tools, timeout=left, **params)
        except TimeoutError:
            raise TimeUp from None
        dropped_from = same if same < len(self._last_messages) else None  # the last call's messages are not a prefix of this call's
        self._last_messages = snapshot
        cost = call_cost(usage, self._price)
        self.cost_usd += cost
        self.usage["input"] += usage.get("prompt_tokens", 0)
        self.usage["cached"] += usage.get("cached_tokens", 0)
        self.usage["output"] += usage.get("completion_tokens", 0)
        self.usage["calls"] += 1
        self.steps.append({"kind": "llm", "t": round(start - self._t0, 3), "duration_s": round(self._clock() - start, 3),
                           "n_messages": len(snapshot), "dropped_or_changed_from": dropped_from,
                           "new_messages": [_capped_message(m) for m in snapshot[same:]], "reply": _capped_message(reply),
                           "tools": [t.get("function", {}).get("name") for t in tools or []], "params": params, "usage": usage})
        return reply


def load_agent(source: str) -> Callable[[str, Env], Any]:
    namespace: dict[str, Any] = {"__name__": "agent"}
    exec(compile(source, "agent.py", "exec"), namespace)  # noqa: S102 - the candidate harness is the thing under test
    run = namespace.get("run")
    if not callable(run):
        raise TypeError("agent.py must define run(task, env)")
    return run


def run_agent(source: str, task: str, *, shell: ShellBackend, llm: LLMBackend, budget_s: float,
              clock: Callable[[], float] = time.monotonic, price: dict[str, float] | None = None) -> dict[str, Any]:
    """Run one candidate harness on one task; never raises for anything the agent does."""
    env = Env(shell, llm, budget_s, clock, price or GLM_PRICE)
    end, error = "returned", None
    try:
        load_agent(source)(task, env)
    except TimeUp:
        end = "timeout"
    except InfraError as e:
        end, error = "infra", cap(str(e), 1_000)
    except Exception:  # noqa: BLE001 - an agent crash is a result, not an infra failure
        end, error = "error", cap(traceback.format_exc(limit=-8), 3_000)
    return {"task": cap(task), "end": end, "error": error, "steps": env.steps, "usage": env.usage,
            "cost_usd": round(env.cost_usd, 9), "agent_s": round(clock() - env._t0, 3), "budget_s": budget_s}
