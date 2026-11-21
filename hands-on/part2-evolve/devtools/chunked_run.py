"""Advance one real evolve run for a bounded time, then exit; run it again to continue.

    uv run python devtools/chunked_run.py --run-dir runs/recheck/code-audit --strategy code-audit --seconds 330

For shells that allow only short foreground commands (a run takes about 20 minutes). It is the attendee's loop with the
real proposer and the real HF Jobs; only two adapters remember their work in the run folder, so a repeated chunk
replays instead of paying twice:

- an evaluation Job is submitted once per spec (`jobs/<hash of the spec>` holds its id); a later chunk reads the same
  Job's log again, whether it is still running or done. The chunk ends by SIGALRM, which kills this process without
  the `KeyboardInterrupt` path that would cancel the Job;
- a proposal is asked once per (iteration folder, prompt, parent agent.py) and stored in `proposals/<hash>/`.

`Evolver` saves its state after the seed calibration and after each iteration, so a chunk that ends inside an
iteration starts that iteration again and both adapters answer from their stores up to the point it had reached.
Dev tool (not shipped in the wheel).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aidev_evolve import cli, evaljob, proposer, strategy


def spec_key(job_kwargs: dict[str, Any]) -> str:
    """Stable across chunks: the gzip'd variable carries a timestamp, the decoded spec does not."""
    spec = evaljob.decode_spec(job_kwargs["env"][evaljob.SPEC_ENV])
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


class RememberedJobs:
    """HfApi stand-in: `run_job` submits a spec's Job once and returns the same Job afterwards."""

    def __init__(self, api: Any, store: Path):
        self._api, self._store = api, Path(store)
        self._store.mkdir(parents=True, exist_ok=True)

    def run_job(self, **kw: Any) -> Any:
        f = self._store / spec_key(kw)
        if f.exists():
            return SimpleNamespace(id=f.read_text().strip(), url=None)
        job = self._api.run_job(**kw)
        f.write_text(job.id)
        return job

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)


def remembered_propose(real: Any, store: Path) -> Any:
    store = Path(store)

    def propose(backend: str, work: Path, prompt: str, **kw: Any) -> proposer.ProposerResult:
        work = Path(work)
        key = hashlib.sha256(f"{work.name}\n{prompt}\n{(work / 'agent.py').read_text()}".encode()).hexdigest()[:16]
        d = store / key
        if (d / "result.json").exists():
            for name in ("agent.py", "PATCH.md"):
                if (d / name).exists():
                    shutil.copy2(d / name, work / name)
            return proposer.ProposerResult(**json.loads((d / "result.json").read_text()))
        res = real(backend, work, prompt, **kw)
        d.mkdir(parents=True, exist_ok=True)
        for name in ("agent.py", "PATCH.md"):
            if (work / name).exists():
                shutil.copy2(work / name, d / name)
        (d / "result.json").write_text(json.dumps({**asdict(res), "log": ""}))
        return res

    return propose


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--strategy", default=strategy.DEFAULT)
    ap.add_argument("--proposer", default="claude")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--board", default=None, help="post to this board (default: none)")
    ap.add_argument("--seconds", type=int, default=330, help="end this chunk after this long")
    ap.add_argument("--iterations", type=int, default=None, help="stop after this many iterations (the session time box does not see "
                    "the pauses between chunks: pass what fits a 22-min session, 2 for the presets)")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    args = cli.build_parser().parse_args(["run", "--strategy", a.strategy, "--run-dir", str(run_dir), "--proposer", a.proposer, "--model", a.model,
                                          "--resume", *(["--board", a.board] if a.board else ["--no-board"])])
    api, token, user = cli._session()
    cli._session = lambda: (RememberedJobs(api, run_dir / "jobs"), token, user)
    proposer.propose = remembered_propose(proposer.propose, run_dir / "proposals")
    signal.alarm(a.seconds)  # default action: the process ends at once, the Job goes on
    evo = cli.make_evolver(args, run_dir)
    evo.run(iterations=a.iterations, heldout_trials=strategy.HELDOUT_TRIALS)
    signal.alarm(0)
    room = getattr(evo, "board", None)
    if hasattr(room, "flush"):
        room.flush(retries=4)
    print(f"\nCHUNKED RUN DONE: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
