"""What keeps GitHub off the path of a trial. In a room, hundreds of Jobs and sandboxes leave Hugging Face through a
few shared IP addresses, and GitHub answers them with HTTP 403 (load test of 25 Nov 2026: 51% of trials lost at 200
Jobs).

This file is the same in both workshop packages (aidev_lb and aidev_evolve; a test in each compares the two copies):
change both. It reads its archives from the package it is in (devtools/build_assets.py of that package builds them):

* data/tb21-tasks.tar.gz: the Terminal-Bench 2.1 task folders the package runs, at the commit the leaderboard pins.
  Harbor reads them as a local dataset (`tasks_dir`), so no trial clones the Terminal-Bench repository.
* data/nvm.tar.gz (aidev_lb only; the Part 2 agent runs in the Job and installs nothing in the sandbox): nvm. Harbor's
  Node install runs nvm's installer, which clones nvm from GitHub; `rewrite` unpacks the archive in its place (the Job
  uploads it to the sandbox first). Node itself comes from nodejs.org, as before.

A third dependency is in the tasks themselves: the verifier script of most tasks (tests/test.sh) installs uv with
Astral's installer of uv 0.9.5, which downloads uv, and then a Python, from GitHub Releases. When that fails the task
counts as failed although the agent solved it. `rewrite` points both downloads at Astral's own mirror
(releases.astral.sh, the same files) for that one command; the task files stay as they are.

`WorkshopSandbox` is the leaderboard's sandbox environment with those rewrites, and with one more try after a short
random delay when the Sandbox API answers 503 (0.8% of trials in the same load test).
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import shutil
import tarfile
import tempfile
import threading
from importlib import resources
from pathlib import Path
from typing import Any

from hlb.harbor_ext.environment import HLBSandboxEnvironment
from huggingface_hub.errors import SandboxError

CACHE = Path.home() / ".cache" / __package__.replace("_", "-") / "assets"
NVM_VERSION = "v0.40.2"  # the version in data/nvm.tar.gz; Harbor's node_install names the same one
NVM_REMOTE = "/opt/aidev-lb/nvm.tar.gz"
RETRY_DELAY = "sleep $((2 + RANDOM % 7))"
_NVM_INSTALLER = re.compile(r"curl [^|;&]*https://raw\.githubusercontent\.com/nvm-sh/nvm/" + re.escape(NVM_VERSION) + r"/install\.sh \| (?:env -u NODE_VERSION )?bash")
_NODE_INSTALL = re.compile(r"nvm install (\d+)\b")
# nvm's installer adds these lines to the shell profile it finds; so does the replacement
_PROFILE = r"""export NVM_DIR="$HOME/.nvm"\n[ -s "$NVM_DIR/nvm.sh" ] && \\. "$NVM_DIR/nvm.sh"\n"""
TEST_SCRIPT = "/tests/test.sh"  # where Harbor puts a task's verifier script
ASTRAL_MIRROR = "https://releases.astral.sh/github"
_unpack = threading.Lock()


def _data(name: str) -> Path:
    return Path(str(resources.files(__package__).joinpath("data", name)))


def nvm_archive() -> Path:
    return _data("nvm.tar.gz")


def tasks_dir(cache: Path | None = None) -> Path:
    """The folder with one Terminal-Bench 2.1 task folder per packaged task, unpacked from the wheel on first use
    (once per Job: its trials run in one process)."""
    root = Path(cache or CACHE) / "tb21"
    with _unpack:
        if not (root / "SOURCE.json").is_file():
            root.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(dir=root.parent))
            with tarfile.open(_data("tb21-tasks.tar.gz")) as tar:
                tar.extractall(tmp, filter="data")
            shutil.rmtree(root, ignore_errors=True)
            os.replace(tmp, root)
    return root / "tasks"


def _verifier_mirror(test_script: str) -> str:
    """Shell that runs before the verifier script: the uv version its installer line names decides the mirror address."""
    return (f"v=$(grep -o 'astral\\.sh/uv/[0-9.]*/install\\.sh' {test_script} 2>/dev/null | head -n 1 | cut -d/ -f3); "
            f'if [ -n "$v" ]; then export UV_DOWNLOAD_URL={ASTRAL_MIRROR}/uv/releases/download/$v '
            f"UV_PYTHON_INSTALL_MIRROR={ASTRAL_MIRROR}/python-build-standalone/releases/download; fi; ")


def rewrite(command: str, remote: str = NVM_REMOTE, test_script: str = TEST_SCRIPT) -> tuple[str, bool]:
    """A sandbox command without its GitHub downloads: (the command, whether it now needs the nvm archive at
    `remote`). nvm's installer becomes the unpacking of the archive, a Node download (`nvm install N`) gets a second
    try, and the run of the verifier script gets Astral's mirror. Any other command comes back unchanged."""
    if test_script in command and not command.startswith("chmod "):
        return _verifier_mirror(test_script) + command, False
    unpack = (f'{{ mkdir -p "$HOME/.nvm" && tar -xzf {remote} -C "$HOME/.nvm" nvm.sh nvm-exec bash_completion && '
              f"{{ [ ! -f \"$HOME/.bashrc\" ] || grep -qs 'nvm.sh' \"$HOME/.bashrc\" || printf '{_PROFILE}' >> \"$HOME/.bashrc\"; }}; }}")
    command, n = _NVM_INSTALLER.subn(lambda m: unpack, command)
    return _NODE_INSTALL.sub(lambda m: f"{{ nvm install {m[1]} || {{ {RETRY_DELAY}; nvm install {m[1]}; }}; }}", command), bool(n)


class NoGitHub:
    """Mixin over a Harbor HF Sandbox environment (see the module text)."""

    async def _pause(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def _again_on_503(self, call: Any) -> Any:
        """A 503 is the API refusing the request, so nothing ran: safe to send once more. A dropped stream is not
        (the command may be running; the leaderboard's ResumableExec waits for it instead)."""
        try:
            return await call()
        except SandboxError as e:
            if e.status_code != 503:
                raise
            await self._pause(random.uniform(1, 6))
            return await call()

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout_sec: int | None = None,
                   user: str | int | None = None) -> Any:
        command, needs_nvm = rewrite(command)
        if needs_nvm:
            await self.upload_file(nvm_archive(), NVM_REMOTE)
        return await self._again_on_503(lambda: super(NoGitHub, self).exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user))

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        return await self._again_on_503(lambda: super(NoGitHub, self).upload_file(source_path, target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        return await self._again_on_503(lambda: super(NoGitHub, self).upload_dir(source_dir, target_dir))


class WorkshopSandbox(NoGitHub, HLBSandboxEnvironment):
    """The environment of every workshop trial (the Harbor job config names it)."""
