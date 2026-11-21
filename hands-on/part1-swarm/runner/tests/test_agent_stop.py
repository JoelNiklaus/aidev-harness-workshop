"""The pinned Harness Leaderboard (`hlb`) stops an agent whose time is up before the verifier starts.

Harbor ends the agent phase with asyncio.wait_for and then runs the tests in the same sandbox. Before
hlb 0.1.9's fix (leaderboard commit 9821cb9, 2026-11-19) nothing stopped the agent's command there: it
kept calling the model and editing files while the tests ran, and the board re-ran 1,130 such trials.
These tests run hlb's own ResumableExec (the exec path of HLBSandboxEnvironment) against a sandbox
stand-in that runs commands on this machine, with a fake agent that ignores SIGTERM and works in three
places (its own loop, a background child, a child in a session of its own).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

resumable_exec = pytest.importorskip("hlb.harbor_ext.resumable_exec")

RUN_CAP_S = 8  # the stand-in ends any command after this, so a red test cannot hang the suite


class LocalSandbox:
    """Duck-types huggingface_hub.Sandbox.run for hlb's `_run`: argv on this machine."""

    id = "local-test-sandbox"

    def run(self, argv, *, shell=False, cwd=None, env=None, timeout=None, check=False):
        p = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = p.communicate(timeout=min(timeout or RUN_CAP_S, RUN_CAP_S))
            return SimpleNamespace(exit_code=p.returncode, stdout=out, stderr=err, timed_out=False)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            return SimpleNamespace(exit_code=None, stdout=out, stderr=err, timed_out=True)


def _cmdlines() -> dict[int, str]:
    out = {}
    for d in Path("/proc").glob("[0-9]*"):
        try:
            out[int(d.name)] = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
    return out


def _alive(word: str) -> list[int]:
    return [pid for pid, c in _cmdlines().items() if word in c and pid != os.getpid()]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    if not Path("/proc").is_dir():
        pytest.skip("needs /proc (Linux, as HF Sandbox)")
    monkeypatch.setattr(resumable_exec, "EXEC_DIR", str(tmp_path / "exec"))
    word = f"zqagent{uuid.uuid4().hex[:8]}"
    work = tmp_path / "work"
    work.mkdir()
    loop = tmp_path / f"{word}-loop.sh"
    loop.write_text(f"while :; do date +%s%N >> {work}/$1; sleep 0.05; done\n")
    script = tmp_path / f"{word}.sh"
    script.write_text(f"trap '' TERM HUP INT\nbash {loop} bg &\nsetsid bash {loop} sid &\nbash {loop} fg\n")
    yield SimpleNamespace(word=word, work=work, command=f"bash {script}")
    for pid in _alive(word):  # never leave a runaway fake agent behind a red test
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _sizes(work: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in work.iterdir()}


def test_an_agent_whose_time_is_up_stops_before_the_verifier(agent):
    rexec = resumable_exec.ResumableExec(lambda _id: None, frozenset(), logging.getLogger("test"))

    async def agent_phase():
        await asyncio.wait_for(rexec.run(LocalSandbox(), agent.command, cwd=None, env=None, timeout_sec=None), timeout=1.5)

    with pytest.raises(TimeoutError):
        asyncio.run(agent_phase())
    # The verifier would start here.
    files = _sizes(agent.work)
    assert set(files) == {"bg", "sid", "fg"}  # it was working in all three places
    time.sleep(1.0)
    assert _sizes(agent.work) == files, "the agent wrote files after its time was up"
    assert _alive(agent.word) == []


def test_no_long_lived_helper_shell_carries_the_agents_command(agent):
    """An agent's `pkill -f <word of its task>` must not end its own run (hlb fix of 2026-11-18): only the
    agent's own processes may carry the command's text, never hlb's wrapper shells."""
    rexec = resumable_exec.ResumableExec(lambda _id: None, frozenset(), logging.getLogger("test"))
    seen: list[str] = []

    async def agent_phase():
        task = asyncio.ensure_future(rexec.run(LocalSandbox(), agent.command, cwd=None, env=None, timeout_sec=None))
        await asyncio.sleep(0.8)
        seen.extend(c for c in _cmdlines().values() if agent.word in c)
        await asyncio.wait_for(task, timeout=0.5)

    with pytest.raises(TimeoutError):
        asyncio.run(agent_phase())
    assert seen, "the fake agent never started"
    assert [c for c in seen if "hlb-" in c] == []
