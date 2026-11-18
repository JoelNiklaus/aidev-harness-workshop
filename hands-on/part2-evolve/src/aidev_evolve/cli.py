"""aidev-evolve command line.

- `aidev-evolve run`: the evolve loop (seed -> propose -> evaluate -> keep or drop), see loop.py.
- `aidev-evolve replay`: play a recorded loop offline (no network).
- `aidev-evolve eval --agent agent.py --tasks a,b --trials K`: score one agent.py once (one HF Job).
- `aidev-evolve eval-job`: what the HF Job runs (reads its spec from the environment).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from aidev_evolve import __version__, board, launch, loop


def _bundled_budget() -> float:
    return float(loop.default_splits()["budget_s"])


def _session() -> tuple[Any, str, str]:
    """HfApi, token and username of the logged-in attendee (`hf auth login` or HF_TOKEN)."""
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token: run `uvx hf auth login` (or set HF_TOKEN).")
    api = HfApi(token=token)
    return api, token, api.whoami()["name"]


def _progress(rec: dict[str, Any]) -> None:
    from aidev_evolve import show

    print(show.trial_line(rec, "eval", sys.stdout.isatty()), end="", flush=True)


def cmd_eval(args: argparse.Namespace) -> int:
    api, token, user = _session()
    source = Path(args.agent).read_text()
    spec = {"candidates": {"cand": source}, "runs": [{"candidate": "cand", "tasks": args.tasks.split(","), "trials": args.trials}], "budget_s": args.budget_s}
    res = launch.run_eval(api, token, user, spec, on_record=_progress, package_url=args.package_url)
    with open(args.out, "a") as f:
        for r in res.records:
            f.write(json.dumps(r) + "\n")
    passed = sum(r["outcome"] == "pass" for r in res.records)
    scored = sum(r["outcome"] != "infra" for r in res.records)
    cost = sum(r["cost_usd"] for r in res.records) + res.job_usd
    print(f"{passed}/{scored} passed, {res.seconds:.0f} s, ${cost:.3f} (Job {res.job_id})")
    return 0


def cmd_eval_job(args: argparse.Namespace) -> int:
    from aidev_evolve import evaljob

    return evaljob.cli()


def make_evolver(args: argparse.Namespace, run_dir: Path, write: Any = None) -> Any:
    """The real adapters: HF Job evaluation in the user's namespace, their coding agent as proposer."""
    from aidev_evolve import proposer

    api, token, user = _session()

    def evaluate(spec: dict[str, Any], on_record: Any) -> Any:
        return launch.run_eval(api, token, user, spec, on_record=on_record, package_url=args.package_url)

    def propose(work: Path, prompt: str) -> Any:
        return proposer.propose(args.proposer, work, prompt, model=args.model, timeout_s=args.proposer_timeout_s)

    room = board.Board(args.board, token) if args.board else None
    extra: dict[str, Any] = {}
    if args.seed:
        extra["seed_source"] = Path(args.seed).read_text()
    splits = json.loads(Path(args.splits).read_text()) if args.splits else loop.default_splits()
    budget_s = args.budget_s or splits["budget_s"]  # the budget the splits' tasks were selected for
    return loop.Evolver(run_dir, splits=splits, evaluate=evaluate, propose=propose, board=room, budget_s=budget_s, color=sys.stdout.isatty(),
                        proposer_name=args.proposer, run_id=f"{user}-{run_dir.name}", write=write or (lambda s: print(s, end="", flush=True)),
                        resume=getattr(args, "resume", False), val_trials=args.val_trials, z=args.z, confirm=args.confirm, **extra)


def _run_dir(base: str | None) -> Path:
    return Path(base) if base else Path("runs") / time.strftime("%Y%m%d-%H%M%S")


def cmd_run(args: argparse.Namespace) -> int:
    evo = make_evolver(args, _run_dir(args.run_dir))
    evo.run(iterations=args.iterations, heldout_trials=args.heldout_trials)
    print(f"\nRun folder: {evo.run_dir}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Phase 2 entry point: R independent runs of the same loop, each with its held-out check; one summary."""
    base = _run_dir(args.run_dir)
    summary = []
    for k in range(args.runs):
        evo = make_evolver(args, base / f"run{k + 1}")
        st = evo.run(iterations=args.iterations, heldout_trials=args.heldout_trials)
        summary.append({"run": k + 1, "incumbent": st["incumbent"], "kept": sum(h["verdict"] == "kept" for h in st["history"]),
                        "val_curve": st["curve"], "heldout": st["heldout"], "spend_usd": st["spend_usd"], "proposer_usd": st["proposer_usd"]})
        (base / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from aidev_evolve import replay

    replay.replay(Path(args.run_dir) if args.run_dir else replay.bundled(), speed=args.speed)
    return 0


def _loop_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--proposer", choices=["claude", "codex", "manual"], default="claude")
    p.add_argument("--model", default=None, help="proposer model (claude: sonnet, opus; codex: its model names)")
    p.add_argument("--proposer-timeout-s", type=float, default=300)
    p.add_argument("--iterations", type=int, default=loop.ITERATIONS)
    p.add_argument("--val-trials", type=int, default=loop.VAL_TRIALS, help="trials per validation task for child and parent each iteration")
    p.add_argument("--heldout-trials", type=int, default=loop.HELDOUT_TRIALS, help="0 skips the final held-out check")
    p.add_argument("--budget-s", type=float, default=None, help="agent time per task (default: the budget the splits were selected for)")
    p.add_argument("--z", type=float, default=loop.Z, help="keep a patch if it wins by z paired standard errors (at least one task)")
    p.add_argument("--no-confirm", dest="confirm", action="store_false", help="keep a winning patch without re-measuring it")
    p.add_argument("--seed", default=None, help="starting agent.py (default: the bundled seed)")
    p.add_argument("--splits", default=None, help="splits JSON (default: the bundled splits)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true", help="continue the run in --run-dir after a crash or Ctrl-C (iterations = total)")
    p.add_argument("--board", default=None, help=f"room board URL to post your progress to (the coordinator Space, {board.ROOM_URL}, once it serves /evolve)")
    p.add_argument("--package-url", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aidev-evolve")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="score one agent.py on some tasks (one HF Job)")
    e.add_argument("--agent", required=True)
    e.add_argument("--tasks", required=True)
    e.add_argument("--trials", type=int, default=1)
    e.add_argument("--budget-s", type=float, default=_bundled_budget())
    e.add_argument("--out", default="records.jsonl")
    e.add_argument("--package-url", default=None)
    e.set_defaults(fn=cmd_eval)
    j = sub.add_parser("eval-job", help="(inside the HF Job) run the spec from $AIDEV_EVOLVE_SPEC")
    j.set_defaults(fn=cmd_eval_job)
    r = sub.add_parser("run", help="evolve the agent: propose, evaluate, keep or drop, repeat")
    _loop_args(r)
    r.set_defaults(fn=cmd_run)
    x = sub.add_parser("experiment", help="several independent runs with held-out checks (Phase 2)")
    _loop_args(x)
    x.add_argument("--runs", type=int, default=3)
    x.set_defaults(fn=cmd_experiment)
    rp = sub.add_parser("replay", help="play a recorded run (offline, no credits)")
    rp.add_argument("run_dir", nargs="?", default=None)
    rp.add_argument("--speed", type=float, default=20.0)
    rp.set_defaults(fn=cmd_replay)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
