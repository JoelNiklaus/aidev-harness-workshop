"""The attendee's working folder: agent.py, its measured versions, their traces and the chart.

In the cloned repo this is part2-harness (agent.py, CONTRACT.md, hints/ and the empty chart.html are files
of the repo). `eval` adds:

    DIR/versions.json         one entry per version: score, outcomes per task
    DIR/versions/vN/agent.py  the file as it was evaluated
    DIR/traces/vN/            INDEX.md and one readable transcript per validation trial (digest.py)
    DIR/chart.html            score by version (chart.py)

Nothing is measured before the attendee's first `eval`: it records version 1. Held-out tasks never get traces: nobody
may read them while fixing.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

from aidev_evolve import chart, digest, runtime

VERSIONS = "versions.json"


class WorkspaceError(Exception):
    """The folder is not usable as asked; the message says what to do."""


def splits() -> dict[str, Any]:
    """The fixed task splits and time budgets (data/splits.json)."""
    return json.loads(resources.files("aidev_evolve.data").joinpath("splits.json").read_text())


def eval_budget_s() -> float:
    """Agent time per task in an attendee evaluation."""
    return float(splits()["eval_budget_s"])


def sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _write(path: Path, text: str) -> None:
    """Replace a file in one step, so a page that reloads itself never reads half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load(directory: Path) -> list[dict[str, Any]]:
    """The measured versions of the folder's agent.py, oldest first (none before the first `eval`)."""
    directory = Path(directory)
    if not (directory / "agent.py").is_file():
        raise WorkspaceError(f"There is no agent.py in {directory.resolve()}: run `aidev-evolve eval` inside part2-harness of the cloned repo.")
    path = directory / VERSIONS
    return json.loads(path.read_text())["versions"] if path.is_file() else []


def _save(directory: Path, versions: list[dict[str, Any]]) -> None:
    _write(directory / VERSIONS, json.dumps({"versions": versions}, indent=1) + "\n")
    _write(directory / "chart.html", chart.render(versions))


def check_agent(source: str) -> None:
    """Refuse a file that cannot be evaluated, before a Job is paid for."""
    try:
        runtime.load_agent(source)
    except SyntaxError as e:
        raise WorkspaceError(f"agent.py does not compile: SyntaxError: {e.msg} (line {e.lineno}). Fix it, then run `aidev-evolve eval` again.") from None
    except TypeError:
        raise WorkspaceError("agent.py must define run(task, env) (see CONTRACT.md).") from None
    except Exception as e:  # noqa: BLE001 - whatever the module's top level raises
        raise WorkspaceError(f"agent.py fails when it is loaded: {type(e).__name__}: {e}") from None


def next_version(versions: list[dict[str, Any]], source: str) -> int | None:
    """The version number an evaluation of `source` gets: 1 in a fresh folder, else the next one; None when the file
    is the latest version, already measured."""
    if not versions:
        return 1
    return None if sha(source) == versions[-1]["sha256"] else versions[-1]["version"] + 1


def version_source(directory: Path, version: int) -> str:
    """agent.py as it was evaluated as this version."""
    path = Path(directory) / "versions" / f"v{version}" / "agent.py"
    if not path.is_file():
        raise WorkspaceError(f"versions/v{version}/agent.py is missing: the held-out check needs the file that was evaluated as v{version}.")
    return path.read_text()


def score_text(v: dict[str, Any]) -> str:
    return f"{v['passed']}/{v['scored']} passed"


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    scored = [r for r in records if r["outcome"] != "infra"]
    return {"passed": sum(r["outcome"] == "pass" for r in scored), "scored": len(scored)}


def record(directory: Path, version: int, source: str, records: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Store an evaluation of agent.py as this version: the evaluated file, its traces, the version list, the chart."""
    directory = Path(directory)
    versions = load(directory)
    entry = {"version": version, "sha256": sha(source), **_counts(records), "infra": sum(r["outcome"] == "infra" for r in records),
             "outcomes": {r["task"]: r["outcome"] for r in records}, **extra}
    vdir = directory / "versions" / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "agent.py").write_text(source)
    digest.write([{**r, "candidate": f"v{version}"} for r in records], directory / "traces" / f"v{version}",
                 title=f"Version {version} of agent.py on the {len(records)} validation tasks: {score_text(entry)}")
    _save(directory, [*versions, entry])
    return entry


def record_heldout(directory: Path, by_version: dict[int, list[dict[str, Any]]], *, trials: int) -> dict[int, dict[str, Any]]:
    """Store the held-out scores of these versions (counts only: held-out transcripts are never written). A version
    without one scored trial gets none."""
    directory = Path(directory)
    versions = load(directory)
    held = {n: {**_counts(records), "trials": trials} for n, records in by_version.items()}
    held = {n: h for n, h in held.items() if h["scored"]}
    for v in versions:
        if v["version"] in held:
            v["heldout"] = held[v["version"]]
    _save(directory, versions)
    return held
