"""Deploy the coordinator as a public Docker Space (results in a PRIVATE bucket), both in the workshop org.

    HF_TOKEN=<an admin token of the org> uv run python -m swarm_coordinator.deploy [--hardware cpu-upgrade] [--target staging]

Stages coordinator/ (code, lock), config/, replay/{reference,evolve-recorded}.json and the runner's README and agent prompt
(for /join) into a temp folder with a Dockerfile
and the Space README, creates the bucket (private) and the Space (public: board and /join, Joel 18 Nov) if
missing (an existing Space keeps its visibility; the script refuses a public bucket), sets the Space secret HF_TOKEN to the scoped
org token in `space-token.txt` (`SPACE_HF_TOKEN=`, gitignored; read here, never printed) and SWARM_BUCKET as a
variable, requests the hardware, and uploads. Rerun after grid.json changes. The upload rebuilds a running Space; a PAUSED
Space stays paused (and does not bill) until `--wake` or `restart_space`. `--target staging` deploys the same code to
agent-harness-workshop/aidev-swarm-staging (bucket ...-staging-results) and wakes it; prod is the default.
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
# prod = the workshop board; staging = the same code for dry runs (Fri 20 Nov with colleagues), its own private bucket
TARGETS = {"prod": (DEFAULT_SPACE, DEFAULT_BUCKET),
           "staging": ("agent-harness-workshop/aidev-swarm-staging", "agent-harness-workshop/aidev-swarm-staging-results")}
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


STAGING_NOTE = """
**STAGING copy** for dry runs: same code as the workshop Space, its own private results bucket. Point both hands-ons
here with `export AIDEV_BOARD=https://agent-harness-workshop-aidev-swarm-staging.hf.space`; empty it afterwards with
`python -m swarm_coordinator.reset`.
"""


def stage(out: Path, *, staging: bool = False) -> Path:
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
    (out / "README.md").write_text(README.replace("title: AI Dev NYC workshop boards", "title: AI Dev NYC workshop boards (staging)")
                                   + STAGING_NOTE if staging else README)
    return out


def deploy(space: str, bucket: str, *, hardware: str = "cpu-basic", api: Any = None, token_file: Path = TOKEN_FILE,
           staging: bool = False, wake: bool = False) -> str:
    """Upload the app. A paused Space stays paused after the upload (it does not rebuild); `wake` restarts it."""
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
        staged = stage(Path(tmp) / "space", staging=staging)
        commit = api.upload_folder(repo_id=space, repo_type="space", folder_path=str(staged), commit_message="Deploy coordinator",
                                   delete_patterns=["*"])
    if wake and getattr(api.get_space_runtime(space), "stage", None) == "PAUSED":
        api.restart_space(space)
    return commit.oid


def add_target_args(ap: argparse.ArgumentParser) -> None:
    """--target prod|staging plus --space/--bucket overrides (deploy and the live check scripts)."""
    ap.add_argument("--target", choices=sorted(TARGETS), default="prod", help="prod: the workshop board; staging: the dry-run copy")
    ap.add_argument("--space", help="override the target's Space")
    ap.add_argument("--bucket", help="override the target's results bucket")


def resolve_target(a: argparse.Namespace) -> argparse.Namespace:
    a.space, a.bucket = a.space or TARGETS[a.target][0], a.bucket or TARGETS[a.target][1]
    return a


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_target_args(ap)
    ap.add_argument("--hardware", default="cpu-basic")
    ap.add_argument("--wake", action="store_true", help="restart the Space when it is paused (staging always is woken)")
    a = resolve_target(ap.parse_args(argv))
    staging = a.target == "staging"
    print("deployed commit", deploy(a.space, a.bucket, hardware=a.hardware, staging=staging, wake=a.wake or staging), "->", space_url(a.space))


if __name__ == "__main__":
    main()
