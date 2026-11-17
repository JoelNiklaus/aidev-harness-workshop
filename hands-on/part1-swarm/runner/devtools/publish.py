"""Build the runner wheel and attach it to the public GitHub repo's release for this version.

    uv run python devtools/publish.py            # builds dist/aidev_swarm-<version>-py3-none-any.whl, then
                                                 # creates release swarm-v<version> (or replaces its wheel)

Attendees and the launched HF Jobs download the wheel from that release without any credentials. Needs
`gh` logged in with push rights on the repo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER / "src"))

from aidev_swarm import __version__, launch  # noqa: E402


def _gh(args: Sequence[str]) -> int:
    return subprocess.run(["gh", *args], check=False).returncode


def publish(wheel: Path, version: str, repo: str, *, gh: Callable[[Sequence[str]], int] = _gh) -> str:
    tag = launch.release_tag(version)
    if gh(["release", "view", tag, "--repo", repo]) == 0:
        rc = gh(["release", "upload", tag, str(wheel), "--clobber", "--repo", repo])
    else:
        rc = gh(["release", "create", tag, str(wheel), "--repo", repo, "--title", f"aidev-swarm {version}",
                 "--notes", f"Runner wheel for the AI Dev NYC swarm (aidev-swarm {version})."])
    if rc != 0:
        raise SystemExit(f"gh failed ({rc}) to publish {wheel.name} to {repo} {tag}")
    return f"https://github.com/{repo}/releases/download/{tag}/{wheel.name}"


def build() -> Path:
    out = RUNNER / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=RUNNER, check=True)
    wheel = out / f"aidev_swarm-{__version__}-py3-none-any.whl"
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
