"""The proposer: the attendee's own coding agent reads the traces and edits agent.py.

Backends: `claude` (Claude Code `claude -p`, the attendee's subscription), `codex` (`codex exec`), and
`manual` (the loop pauses; paste PROMPT.md into any coding agent, or edit agent.py yourself, then press
Enter). The proposer works in a folder holding only agent.py, the traces of the current agent on the
search tasks, the history of earlier attempts and the contract; it may read and edit files there and
nothing else (no shell: the controller, not the proposer, evaluates). It explains its patch in PATCH.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMEOUT_S = 300
CLAUDE_TOOLS = "Read,Edit,Write,Glob,Grep"
STRIP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "HF_TOKEN")

HEADER = """You are improving a small coding-agent harness: the Python file `agent.py` in this folder.
A fixed model (GLM-5.3-Flash) runs inside this harness on Terminal-Bench 2.1 tasks: Linux terminal tasks
checked by hidden tests. Each task run gets {budget_s:.0f} seconds; the model, its settings and the time
budget are fixed. Only the harness can change: prompts, the loop, tools, how outputs are shown to the
model, when to stop, how to recover from mistakes.

This is iteration {iteration}.
"""

# The fixed part of every proposer prompt: a strategy's prompt file cannot remove or change it.
RULES = """
Rules (fixed for everyone):
- General harness behaviour only: never mention benchmark task names or task-specific content.
- Keep the contract: `run(task, env)`; only `env.llm` and `env.shell` reach the model and the container;
  standard-library imports from: json, re, time, textwrap, shlex, collections, dataclasses, itertools,
  functools, math, random, string, typing, base64, hashlib, difflib, enum, datetime. No open/eval/exec.
- Keep agent.py short and readable (it is shown on a projector): prefer a few clear lines over a framework.
- Do not run anything; you cannot run the benchmark. Work fast (about 2 minutes).

Finally write `PATCH.md`: first line `# <title, at most 8 words>`, then three short lines:
`Hypothesis:` (which failure, seen in which transcripts), `Change:` (what you changed), `Expected:` (which
failures it should fix, and what it could break).
"""

CONTRACT = """# The contract between agent.py and the evaluation

`run(task, env)` is called once per task, with the task text. `env` offers:

- `env.llm(messages, tools=None, *, temperature=None, max_tokens=None, top_p=None)` -> the assistant message
  as a dict: `role`, `content` (str or None) and, when the model calls tools, `tool_calls`
  (`[{"id", "type": "function", "function": {"name", "arguments": <JSON string>}}]`). OpenAI chat format:
  answer each tool call with `{"role": "tool", "tool_call_id": <id>, "content": <str>}`.
  The model (GLM-5.3-Flash), provider and reasoning effort are fixed.
- `env.shell(command, timeout=None)` -> object with `exit_code`, `output` (stdout and stderr together) and
  `timed_out`. Each call is a fresh `bash -c` in the task container (the working directory and variables do
  not carry over). The sandbox kills the command's whole process group at its timeout; without a timeout
  a command may run until the task's time budget is gone, which ends the run.
- `env.time_left()` -> seconds left of the budget. `env.log(text)` adds a note to the transcript.

When `run` returns (or the budget is gone, or it raises), the task's hidden tests check the container.
"""


@dataclass
class ProposerResult:
    ok: bool
    seconds: float
    cost_usd: float | None = None
    error: str | None = None
    log: str = ""


def build_prompt(*, iteration: int, budget_s: float, approach: str | None = None, hint: str = "") -> str:
    """The fixed header, the strategy's approach (its proposer_prompt.md; default: the default preset), an optional
    hint for a second candidate, then the fixed rules."""
    if approach is None:
        from aidev_evolve import strategy

        approach = strategy.load(strategy.DEFAULT).prompt
    extra = f"\n{hint.strip()}\n" if hint.strip() else ""
    return HEADER.format(iteration=iteration, budget_s=budget_s) + "\n" + approach.strip() + "\n" + extra + RULES


def _claude_json(out: str) -> dict[str, Any] | None:
    """The result object of `claude -p --output-format json`, ignoring warning lines the CLI prints around it."""
    try:
        data = json.loads(out)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(out.splitlines()):
        if line.lstrip().startswith("{"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "result" in data:
                return data
    return None


def _codex_answer(out: str) -> str:
    """The agent's reply in `codex exec` output: after the last `codex` header line, before the `tokens used` footer."""
    lines = out.splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == "codex"]
    body = lines[starts[-1] + 1:] if starts else lines
    end = next((i for i, l in enumerate(body) if l.strip() == "tokens used"), len(body))
    return "\n".join(body[:end]).strip()


def _run(argv: list[str], cwd: Path, env: dict[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)


def propose(backend: str, work: Path, prompt: str, *, model: str | None = None, timeout_s: float = TIMEOUT_S,
            run: Callable[..., Any] = _run, which: Callable[[str], str | None] = shutil.which,
            ask: Callable[[str], str] = input, clock: Callable[[], float] = time.monotonic) -> ProposerResult:
    start = clock()
    work = Path(work)
    if backend == "manual":
        ask(f"\n  Paste {work / 'PROMPT.md'} into your coding agent (or edit {work / 'agent.py'} yourself and write PATCH.md),\n"
            f"  then press Enter here to evaluate the change... ")
        return ProposerResult(True, clock() - start)
    exe = which(backend)
    if not exe:
        return ProposerResult(False, 0.0, error=f"`{backend}` not found on PATH: install and log in at home, or use --proposer manual")
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    if backend == "claude":
        argv = [exe, "-p", prompt, "--permission-mode", "acceptEdits", "--allowedTools", CLAUDE_TOOLS, "--setting-sources", "",
                "--output-format", "json", *(["--model", model] if model else [])]
    elif backend == "codex":
        argv = [exe, "exec", "--sandbox", "workspace-write", "--skip-git-repo-check", "-C", str(work), *(["-m", model] if model else []), prompt]
    else:
        raise ValueError(f"unknown proposer {backend!r} (claude, codex, manual)")
    try:
        cp = run(argv, cwd=work, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return ProposerResult(False, clock() - start, error=f"proposer still running after {timeout_s:.0f} s: stopped")
    seconds = clock() - start
    out = cp.stdout or ""
    if backend == "claude":
        data = _claude_json(out)
        if data is None:
            return ProposerResult(False, seconds, error=f"claude exited {cp.returncode}: {(out or cp.stderr or '')[-400:]}", log=out)
        if data.get("is_error") or cp.returncode != 0:
            return ProposerResult(False, seconds, data.get("total_cost_usd"), error=str(data.get("result"))[:400], log=out)
        return ProposerResult(True, seconds, data.get("total_cost_usd"), log=out)
    if cp.returncode != 0:
        return ProposerResult(False, seconds, error=f"codex exited {cp.returncode}: {(cp.stderr or out)[-400:]}", log=out)
    return ProposerResult(True, seconds, log=out)


PROBE_PROMPT = "Reply with the single word: ok"
LOGIN_FIX = {"claude": "run `claude` and type /login", "codex": "run `codex login`"}


def probe(backend: str, *, run: Callable[..., Any] = _run, which: Callable[[str], str | None] = shutil.which,
          timeout_s: float = 120) -> tuple[bool, str]:
    """One headless call of the attendee's coding agent (the same way the loop calls it): (ok, one-line message)."""
    exe = which(backend)
    if not exe:
        return False, f"{backend}: not found on PATH (install it, open a new terminal, {LOGIN_FIX[backend]})"
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    argv = ([exe, "-p", PROBE_PROMPT, "--output-format", "json", "--setting-sources", ""] if backend == "claude"
            else [exe, "exec", "--skip-git-repo-check", PROBE_PROMPT])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cp = run(argv, cwd=Path(tmp), env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"{backend}: no answer after {timeout_s:.0f} s (network, or {LOGIN_FIX[backend]})"
    out = cp.stdout or ""
    if backend == "claude":
        data = _claude_json(out)
        answer, failed = (str(data.get("result", "")), bool(data.get("is_error"))) if data else ((out or cp.stderr or "")[-200:], True)
    else:
        last_err = next((l for l in reversed((cp.stderr or "").splitlines()) if l.strip()), "")
        answer, failed = _codex_answer(out) or last_err, False
    said_ok = any(l.strip().strip(".!").lower() == "ok" for l in answer.splitlines())
    if failed or cp.returncode != 0 or not said_ok:
        return False, f"{backend}: {' '.join(answer.split())[:160] or f'exit {cp.returncode}'} ({LOGIN_FIX[backend]})"
    return True, f"{backend}: answers headless calls"


def read_patch_notes(work: Path) -> tuple[str, str]:
    p = Path(work) / "PATCH.md"
    if not p.exists():
        return "(no PATCH.md)", ""
    text = p.read_text().strip()
    first = text.splitlines()[0] if text else ""
    return (first.lstrip("# ").strip() or "(untitled)"), text
