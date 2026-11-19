"""aidev-evolve command line.

- `aidev-evolve run --strategy <preset|folder>`: the evolve loop (seed -> propose -> evaluate -> keep or drop), see loop.py.
- `aidev-evolve strategies` / `strategy-init <preset> <dir>`: the optimization strategies (strategy.py).
- `aidev-evolve rank <run folders>`: rank strategies by held-out gain with CIs and ties (ranking.py).
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

from aidev_evolve import __version__, board, launch, loop, ranking, strategy


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


def _http() -> Any:
    import httpx

    return httpx.Client(timeout=10.0)


def cmd_check(args: argparse.Namespace) -> int:
    """Everything `run` needs, in seconds and for free (`--job` adds one real HF Job: about a minute, under a cent)."""
    from aidev_evolve import proposer

    print(f"aidev-evolve {__version__}: checking your setup")

    def line(ok: bool | None, text: str) -> None:
        print(f"  {'ok  ' if ok else ('FAIL' if ok is False else 'warn')}  {text}", flush=True)

    session = None
    try:
        session = _session()
        line(True, f"Hugging Face: logged in as {session[2]}")
    except SystemExit as e:
        line(False, f"Hugging Face: {e}")
    except Exception as e:  # noqa: BLE001 - a bad token, no network
        line(False, f"Hugging Face: {type(e).__name__}: {e} (run `uvx hf auth login` again)")
    failed = session is None  # go on: show every problem in one go
    backends = [args.proposer] if args.proposer else ["claude", "codex"]
    working = []
    for b in backends:
        ok, msg = proposer.probe(b)
        line(ok if (ok or args.proposer) else None, msg)
        if ok:
            working.append(b)
    if not working:
        failed = True
        print("  No coding agent answered. Log in (see above), or run the loop with `--proposer manual` and paste each prompt into any agent.")
    try:
        r = _http().get(f"{args.board.rstrip('/')}/api/evolve/state")
        line(True if r.status_code == 200 else None,
             f"workshop board {board.page_url(args.board)}: " + ("reachable" if r.status_code == 200 else f"HTTP {r.status_code} (not live yet; your loop runs without it)"))
    except Exception as e:  # noqa: BLE001 - the board is a display, never a blocker
        line(None, f"workshop board {board.page_url(args.board)}: {type(e).__name__} (your loop runs without it and sends its updates later)")
    if args.job and session:
        from aidev_evolve import loop, providers

        api, token, user = session
        order = providers.for_user(user)
        line(True, f"GLM provider: {order[0]} (fallback: {', '.join(order[1:]) or 'none'})")
        spec = {"candidates": {"seed": loop.SEED_SOURCE}, "runs": [{"candidate": "seed", "tasks": ["fix-git"], "trials": 1}], "budget_s": _bundled_budget(),
                "providers": order}
        try:
            res = launch.run_eval(api, token, user, spec, on_record=_progress, package_url=args.package_url)
            outcome = res.records[0]["outcome"] if res.records else "no result"
            cost = sum(r.get("cost_usd") or 0.0 for r in res.records) + res.job_usd
            line(outcome != "infra" and bool(res.records), f"HF Job {res.job_id}: seed agent on one task, {outcome}, {res.seconds:.0f} s, ${cost:.4f}")
            failed |= outcome == "infra" or not res.records
        except Exception as e:  # noqa: BLE001
            line(False, f"HF Job: {type(e).__name__}: {str(e)[:200]} (credits: open the coupon link; 403: the token needs the Jobs permission)")
            failed = True
    if failed:
        return 1
    print(f"ready: start your loop with `aidev-evolve run --proposer {working[0]}`")
    return 0


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
    from aidev_evolve import proposer, providers

    api, token, user = _session()

    def evaluate(spec: dict[str, Any], on_record: Any) -> Any:
        return launch.run_eval(api, token, user, spec, on_record=on_record, package_url=args.package_url)

    def propose(work: Path, prompt: str) -> Any:
        return proposer.propose(args.proposer, work, prompt, model=args.model, timeout_s=args.proposer_timeout_s)

    room = board.QueuedBoard(board.Board(args.board, token), run_dir / "outbox.jsonl") if args.board else None
    extra: dict[str, Any] = {}
    if args.seed:
        extra["seed_source"] = Path(args.seed).read_text()
    splits = json.loads(Path(args.splits).read_text()) if args.splits else loop.default_splits()
    budget_s = args.budget_s or splits["budget_s"]  # the budget the splits' tasks were selected for
    session_s = None if getattr(args, "no_time_box", False) else strategy.SESSION_MINUTES * 60
    return loop.Evolver(run_dir, splits=splits, evaluate=evaluate, propose=propose, board=room, budget_s=budget_s, color=sys.stdout.isatty(),
                        proposer_name=args.proposer, run_id=f"{user}-{run_dir.name}", write=write or (lambda s: print(s, end="", flush=True)),
                        resume=getattr(args, "resume", False), strategy=strategy.load(args.strategy), session_s=session_s,
                        parallel_proposals=args.proposer != "manual", providers=providers.for_user(user), **extra)


def _run_dir(base: str | None) -> Path:
    return Path(base) if base else Path("runs") / time.strftime("%Y%m%d-%H%M%S")


def _strategy_or_exit(spec: str) -> strategy.Strategy | None:
    try:
        return strategy.load(spec)
    except strategy.StrategyError as e:
        print(f"Strategy {spec!r}: {e}")
        return None


def cmd_run(args: argparse.Namespace) -> int:
    from aidev_evolve import proposer

    s = _strategy_or_exit(args.strategy)  # before any login check or paid call
    if s is None:
        return 2
    print("Strategy " + strategy.describe(s), flush=True)
    if args.proposer != "manual":  # the seed calibration costs credits: a logged-out proposer must stop us before it
        ok, msg = proposer.probe(args.proposer)
        if not ok:
            print(f"Your coding agent does not answer: {msg}\nFix it (`aidev-evolve check` shows every step), or use --proposer manual.")
            return 1
    evo = make_evolver(args, _run_dir(args.run_dir))
    if args.board:
        print(f"Workshop board: {board.page_url(args.board)}  (your progress appears there; --no-board keeps it private)", flush=True)
    try:
        evo.run(heldout_trials=strategy.HELDOUT_TRIALS)
    except launch.EvalJobError as e:  # the seed calibration or held-out Job failed: nothing to continue on
        print(f"\nAn evaluation Job failed: {e}\nFix the cause, then `aidev-evolve run --resume --run-dir {evo.run_dir}` "
              f"continues without repeating paid work.")
        return 1
    room = getattr(evo, "board", None)
    left = room.flush(retries=4) if isinstance(room, board.QueuedBoard) else 0
    if left:
        print(f"\nThe workshop board did not take {left} updates; `aidev-evolve run --resume --run-dir {evo.run_dir}` sends them later.")
    print(f"\nRun folder: {evo.run_dir}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Phase 2 entry point: R independent runs of the same loop, each with its held-out check; one summary."""
    if _strategy_or_exit(args.strategy) is None:
        return 2
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


def cmd_strategies(args: argparse.Namespace) -> int:
    print(f"Optimization strategies (fixed for everyone: model, {loop.default_splits()['budget_s']:.0f} s per task, the splits and held-out set, "
          f"the seed, {strategy.HELDOUT_TRIALS} held-out trials, a {strategy.SESSION_MINUTES:.0f}-min session, at most ${strategy.USD_CAP:.0f}):\n")
    for name in strategy.presets():
        print(strategy.describe(strategy.load(name)) + "\n")
    print("Edit one: `aidev-evolve strategy-init <preset> my-strategy`, change strategy.toml and proposer_prompt.md, "
          "then `aidev-evolve run --strategy my-strategy`.")
    return 0


def cmd_strategy_init(args: argparse.Namespace) -> int:
    try:
        d = strategy.init(args.preset, Path(args.directory))
    except (strategy.StrategyError, FileExistsError) as e:
        print(e)
        return 2
    print(f"Copied {args.preset} to {d}/: edit strategy.toml (knobs) and proposer_prompt.md (your proposer's approach),\n"
          f"then run it with `aidev-evolve run --strategy {d}` (the file is checked before anything is spent).")
    return 0


def run_entry(run_dir: Path) -> dict[str, Any]:
    """A run folder as the ranking reads it (the same fields as the board's heldout event)."""
    config = json.loads((run_dir / "config.json").read_text())
    state = json.loads((run_dir / "state.json").read_text())
    rid = state.get("run_id") or config.get("run_id") or run_dir.name
    return {"run_id": rid, "user": rid.split("-")[0], "strategy": config["strategy"], "heldout": state.get("heldout"),
            "kept": [h.get("title") for h in state.get("history", []) if h.get("verdict") == "kept"]}


def cmd_rank(args: argparse.Namespace) -> int:
    runs = [run_entry(Path(d)) for d in args.run_dirs]
    result = ranking.rank(runs, by=args.by, min_runs=args.min_runs)
    print(ranking.format_table(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from aidev_evolve import replay

    print("RECORDED REPLAY — historical run, not live results.", flush=True)
    replay.replay(Path(args.run_dir) if args.run_dir else replay.bundled(), speed=args.speed)
    return 0


def _loop_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--proposer", choices=["claude", "codex", "manual"], default="claude")
    p.add_argument("--model", default=None, help="proposer model (claude: sonnet, opus; codex: its model names)")
    p.add_argument("--proposer-timeout-s", type=float, default=300)
    p.add_argument("--strategy", default=strategy.DEFAULT,
                   help=f"optimization strategy: a preset ({', '.join(strategy.presets())}) or a folder from `strategy-init`")
    p.add_argument("--budget-s", type=float, default=None, help="agent time per task (default: the budget the splits were selected for)")
    p.add_argument("--seed", default=None, help="starting agent.py (default: the bundled seed)")
    p.add_argument("--splits", default=None, help="splits JSON (default: the bundled splits)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true", help="continue the run in --run-dir after a crash or Ctrl-C (iterations = total)")
    p.add_argument("--board", default=board.default_url(),
                   help=f"workshop board to post your progress to (default: $AIDEV_BOARD, else the coordinator Space, {board.ROOM_URL})")
    p.add_argument("--no-board", dest="board", action="store_const", const=None, help="do not post to the workshop board")
    p.add_argument("--package-url", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aidev-evolve")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check your setup: HF login, your coding agent, the workshop board (free, seconds)")
    c.add_argument("--proposer", choices=["claude", "codex"], default=None, help="check only this coding agent (default: both, one must answer)")
    c.add_argument("--board", default=board.default_url(), help="workshop board (default: $AIDEV_BOARD, else the coordinator Space)")
    c.add_argument("--job", action="store_true", help="also run one seed trial in an HF Job (about a minute, under a cent)")
    c.add_argument("--package-url", default=None)
    c.set_defaults(fn=cmd_check)
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
    x.add_argument("--iterations", type=int, default=None, help="(dev) iterations per run (default: the strategy's, inside the time box)")
    x.add_argument("--heldout-trials", type=int, default=strategy.HELDOUT_TRIALS, help="(dev) 0 skips the held-out check")
    x.add_argument("--no-time-box", action="store_true", help="(dev) no session time box")
    x.set_defaults(fn=cmd_experiment)
    room = sub.add_parser("room-init", help="create an offline shared-seed proposal kit")
    room.add_argument("directory")
    def init_room(args: argparse.Namespace) -> int:
        from aidev_evolve import room as room_module
        room_module.init(Path(args.directory))
        return 0
    room.set_defaults(fn=init_room)
    rd = sub.add_parser("room-diff", help="print a proposal as text without executing it")
    rd.add_argument("directory")
    def diff_room(args: argparse.Namespace) -> int:
        from aidev_evolve import room as room_module
        room_module.diff(Path(args.directory))
        return 0
    rd.set_defaults(fn=diff_room)
    st = sub.add_parser("strategies", help="list the optimization strategy presets with their knobs, time and credits")
    st.set_defaults(fn=cmd_strategies)
    si = sub.add_parser("strategy-init", help="copy a strategy preset into a folder to edit it")
    si.add_argument("preset")
    si.add_argument("directory", nargs="?", default="my-strategy")
    si.set_defaults(fn=cmd_strategy_init)
    rk = sub.add_parser("rank", help="rank strategies by held-out gain (95%% CIs, ties) from run folders")
    rk.add_argument("run_dirs", nargs="+")
    rk.add_argument("--by", choices=["hash", "preset", "name"], default="hash", help="pool identical strategies (hash) or every edit of a preset")
    rk.add_argument("--min-runs", type=int, default=ranking.MIN_RUNS)
    rk.add_argument("--json", default=None, help="also write the full ranking as JSON")
    rk.set_defaults(fn=cmd_rank)
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
