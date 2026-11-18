"""Deploy the coordinator as a public Docker Space (results in a PRIVATE bucket), both in the workshop org.

    HF_TOKEN=<an admin token of the org> uv run python -m swarm_coordinator.deploy [--hardware cpu-upgrade]

Stages coordinator/ (code, lock), config/, replay/{reference,evolve-recorded}.json and the runner's README and agent prompt
(for /join) into a temp folder with a Dockerfile
and the Space README, creates the bucket (private) and the Space (public: board and /join, Joel 18 Nov) if
missing (an existing Space keeps its visibility; the script refuses a public bucket), sets the Space secret HF_TOKEN to the scoped
org token in `space-token.txt` (`SPACE_HF_TOKEN=`, gitignored; read here, never printed) and SWARM_BUCKET as a
variable, requests the hardware, and uploads. Rerun after grid.json changes. The upload rebuilds and starts the Space,
so a paused Space wakes up (and bills) after a deploy.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import paths

COORD = Path(__file__).resolve().parents[2]
DEFAULT_SPACE = "agent-harness-workshop/aidev-swarm"
DEFAULT_BUCKET = "agent-harness-workshop/aidev-swarm-results"
TOKEN_FILE = COORD / "space-token.txt"
TOKEN_KEY = "SPACE_HF_TOKEN"


def space_url(space_id: str) -> str:
    """The app host of a Space: https://<owner>-<name>.hf.space (lowercase, '_' and '.' become '-')."""
    return "https://" + space_id.replace("/", "-").replace("_", "-").replace(".", "-").lower() + ".hf.space"


def space_token(path: Path = TOKEN_FILE) -> str:
    """The token the Space writes results with (fine-grained, write on the workshop org only)."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    for line in lines:
        if line.startswith(TOKEN_KEY + "="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit(f"no {TOKEN_KEY}=... in {path}: paste the scoped org token there first")

DOCKERFILE = """FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \\
    SWARM_ROOT=/home/user/app PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user coordinator/pyproject.toml coordinator/uv.lock coordinator/
RUN cd coordinator && uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .
RUN cd coordinator && uv sync --frozen --no-dev
EXPOSE 7860
CMD ["coordinator/.venv/bin/coordinator", "--host", "0.0.0.0", "--port", "7860", "--db", "/tmp/swarm.db"]
"""

README = """---
title: AI Dev NYC workshop boards
emoji: 🐝
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Coordinator and live boards of the AI Dev NYC 2026 Agent Harnesses workshop: Part 1 swarm evaluation (/)
and Part 2 evolve race (/evolve).
API: see PROTOCOL.md in the workshop repo. Results persist to a private HF bucket.
"""


def stage(out: Path) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    # Allowlist, not a denylist: coordinator/ also holds gitignored local files (space-token.txt with the org token).
    (out / "coordinator").mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(COORD / name, out / "coordinator" / name)
    shutil.copytree(COORD / "src", out / "coordinator" / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "config").mkdir(parents=True)
    shutil.copy2(paths.grid_path(), out / "config" / "grid.json")
    shutil.copy2(paths.prices_path(), out / "config" / "prices.json")
    (out / "replay").mkdir()
    shutil.copy2(paths.reference_path(), out / "replay" / "reference.json")
    shutil.copy2(paths.evolve_replay_path(), out / "replay" / "evolve-recorded.json")
    (out / "runner").mkdir()
    for name in ("README.md", "AGENT_PROMPT.md"):  # the /join page's command and prompt
        shutil.copy2(paths.runner_dir() / name, out / "runner" / name)
    (out / "Dockerfile").write_text(DOCKERFILE)
    (out / "README.md").write_text(README)
    return out


def deploy(space: str, bucket: str, *, hardware: str = "cpu-basic", api: Any = None, token_file: Path = TOKEN_FILE) -> str:
    token = space_token(token_file)  # fail before any Hub call
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_bucket(bucket, private=True, exist_ok=True)
    if not api.bucket_info(bucket).private:
        raise SystemExit(f"bucket {bucket} is public; refusing to write results there")
    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True, space_hardware=hardware)
    if api.get_space_runtime(space).requested_hardware != hardware:
        api.request_space_hardware(space, hardware)
    api.add_space_secret(space, "HF_TOKEN", token)
    api.add_space_variable(space, "SWARM_BUCKET", bucket)
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp) / "space")
        commit = api.upload_folder(repo_id=space, repo_type="space", folder_path=str(staged), commit_message="Deploy coordinator",
                                   delete_patterns=["*"])
    return commit.oid


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=DEFAULT_SPACE)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--hardware", default="cpu-basic")
    a = ap.parse_args(argv)
    print("deployed commit", deploy(a.space, a.bucket, hardware=a.hardware), "->", space_url(a.space))


if __name__ == "__main__":
    main()
