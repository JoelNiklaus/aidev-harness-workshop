"""Build the aidev-evolve wheel and attach it to the public GitHub repo's release for this version.

    uv run python devtools/publish.py     # dist/aidev_evolve-<version>-py3-none-any.whl -> release evolve-v<version>

The evaluation Jobs (and attendees) download it from there without any credentials, as in Part 1. Needs `gh`
logged in with push rights on the repo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidev_evolve import __version__, launch  # noqa: E402


def _gh(args: Sequence[str]) -> int:
    return subprocess.run(["gh", *args], check=False).returncode


def publish(wheel: Path, version: str, repo: str, *, gh: Callable[[Sequence[str]], int] = _gh) -> str:
    tag = launch.release_tag(version)
    if gh(["release", "view", tag, "--repo", repo]) == 0:
        rc = gh(["release", "upload", tag, str(wheel), "--clobber", "--repo", repo])
    else:
        rc = gh(["release", "create", tag, str(wheel), "--repo", repo, "--title", f"aidev-evolve {version}",
                 "--notes", f"Part 2 evolve-loop wheel for the AI Dev NYC workshop (aidev-evolve {version})."])
    if rc != 0:
        raise SystemExit(f"gh failed ({rc}) to publish {wheel.name} to {repo} {tag}")
    return f"https://github.com/{repo}/releases/download/{tag}/{wheel.name}"


def build() -> Path:
    out = ROOT / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=ROOT, check=True)
    wheel = out / f"aidev_evolve-{__version__}-py3-none-any.whl"
    if not wheel.exists():
        raise SystemExit(f"uv build did not write {wheel}")
    return wheel


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=launch.GITHUB_REPO)
    a = ap.parse_args(argv)
    print(publish(build(), __version__, a.repo))


if __name__ == "__main__":
    main()
