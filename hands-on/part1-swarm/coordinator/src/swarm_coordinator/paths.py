"""Where the coordinator finds the grid, the prices and the replay data.

Layout (repo and Docker image alike): <root>/config/{grid,prices}.json, <root>/replay/{reference,evolve-recorded}.json,
<root>/runner/{README,AGENT_PROMPT}.md (the /join page), <root>/evolve/README.md (the /evolve/join page; in the repo
the Part 2 package next door, hands-on/part2-evolve/).
<root> is $SWARM_ROOT, else hands-on/part1-swarm/ of this checkout.
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("SWARM_ROOT") or Path(__file__).resolve().parents[3])


def grid_path() -> Path:
    return root() / "config" / "grid.json"


def prices_path() -> Path:
    return root() / "config" / "prices.json"


def reference_path() -> Path:
    return root() / "replay" / "reference.json"


def evolve_replay_path() -> Path:
    return root() / "replay" / "evolve-recorded.json"


def harness_meta_path() -> Path:
    return Path(__file__).with_name("data") / "harnesses.json"


def runner_dir() -> Path:
    return root() / "runner"


def evolve_dir() -> Path:
    staged = root() / "evolve"
    return staged if staged.is_dir() else root().parent / "part2-evolve"
