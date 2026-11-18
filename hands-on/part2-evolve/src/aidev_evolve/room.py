"""Offline room exercise: make and inspect proposals without executing them."""

import difflib
import hashlib
from importlib import resources
from pathlib import Path

from aidev_evolve import proposer


def init(directory: Path) -> None:
    directory = Path(directory)
    try:
        directory.mkdir()
    except FileExistsError:
        raise SystemExit(f"Refusing to overwrite existing path: {directory}") from None
    source = resources.files('aidev_evolve.seed').joinpath('agent.py').read_bytes()
    (directory / 'base.py').write_bytes(source)
    (directory / 'agent.py').write_bytes(source)
    (directory / 'BASE.sha256').write_text(hashlib.sha256(source).hexdigest() + '\n')
    (directory / 'CONTRACT.md').write_text(proposer.CONTRACT)
    (directory / 'README.md').write_text('''# Shared harness exercise: offline proposal\n\nEveryone starts from the same seed and its SHA-256 in BASE.sha256.\nEdit agent.py yourself or with your own coding agent subscription.\nRead CONTRACT.md first. Propose one focused general harness change.\nDescribe the hypothesis and trade-offs to the facilitator.\nThe facilitator reviews contributions as text. Do not execute other attendees' code.\nRun `aidev-evolve room-diff <this-directory>` to print your proposal and base hash.\nShare that output and your hypothesis using the facilitator's chosen channel.\nUse `aidev-evolve replay` to show a recorded real run, not a live score.\nNo submission, live evaluation, or shared champion is provided by this kit.\n''')
    print(f"Room kit created at {directory}. No code executed and no network used.")


def diff(directory: Path) -> None:
    base = (directory / 'base.py').read_bytes()
    expected = (directory / 'BASE.sha256').read_text().strip()
    if hashlib.sha256(base).hexdigest() != expected:
        raise SystemExit('The base does not match BASE.sha256. Create a fresh kit.')
    edited = (directory / 'agent.py').read_text()
    patch = difflib.unified_diff(base.decode().splitlines(keepends=True), edited.splitlines(keepends=True),
                                 fromfile='base.py', tofile='agent.py')
    print(f'Base SHA-256: {expected}')
    print(''.join(patch), end='')
