"""aidev-evolve command line.

- `aidev-evolve eval [--heldout]`: score the folder's agent.py as its next version (one HF Job); trajectories (traces/),
  versions.json, chart.html (workspace.py). The first eval records version 1.
- `aidev-evolve eval-job`: what the HF Job runs (reads its spec from the environment).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from aidev_evolve import __version__, launch

RESET, RED, GREEN, YELLOW = "\033[0m", "\033[31m", "\033[32m", "\033[33m"

def _session() -> tuple[Any, str, str]:
    """HfApi, token and username of the logged-in attendee (`hf auth login` or HF_TOKEN)."""
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token: run `hf auth login` (or set HF_TOKEN).")
    api = HfApi(token=token)
    return api, token, api.whoami()["name"]


def _trial_line(rec: dict[str, Any], color: bool) -> str:
    text, code = {"pass": ("PASS ", GREEN), "fail": ("fail ", RED), "infra": ("INFRA", YELLOW)}[rec["outcome"]]
    mark = f"{code}{text}{RESET}" if color else text
    return f"  {mark} {rec['task']:<28} {rec.get('wall_s') or 0:4.0f} s"


def _pct(h: dict[str, Any]) -> str:
    return f"{h['passed']}/{h['scored']} = {100 * h['passed'] / h['scored']:.0f}%"


def cmd_eval(args: argparse.Namespace) -> int:
    """Score the folder's agent.py as its next version: one HF Job, the validation tasks, one trial each. With
    --heldout: the latest version and version 1 on the held-out tasks, in one Job."""
    from aidev_evolve import providers, workspace

    d = Path(args.dir)
    try:
        versions = workspace.load(d)
        source = (d / "agent.py").read_text()
        workspace.check_agent(source)
    except workspace.WorkspaceError as e:
        print(e)
        return 2
    split = workspace.splits()
    version = workspace.next_version(versions, source)
    again = "aidev-evolve eval" + (" --heldout" if args.heldout else "")
    if args.heldout:
        if version is not None:
            print("agent.py is not a measured version yet: run `aidev-evolve eval` first, then `aidev-evolve eval --heldout`.")
            return 2
        tasks = split["heldout"]
        wanted = [v for v in {versions[0]["version"]: versions[0], versions[-1]["version"]: versions[-1]}.values()]
        todo = [v for v in wanted if not v.get("heldout")]
        if not todo:
            print("The held-out scores are measured already: " + ", ".join(f"v{v['version']}: {_pct(v['heldout'])}" for v in wanted)
                  + ". Change agent.py and run `aidev-evolve eval` to get a new version.")
            return 0
        try:
            candidates = {f"v{v['version']}": source if v is versions[-1] else workspace.version_source(d, v["version"]) for v in todo}
        except workspace.WorkspaceError as e:
            print(e)
            return 2
        what = f"{' and '.join(candidates)} on the {len(tasks)} held-out tasks"
    elif version is None:
        last = versions[-1]
        print(f"agent.py is unchanged since v{last['version']} ({workspace.score_text(last)}): nothing to evaluate. Change agent.py, then run `aidev-evolve eval` again.")
        return 0
    else:
        candidates, tasks = {f"v{version}": source}, split["validation"]
        what = f"agent.py as v{version} on the {len(tasks)} validation tasks"
    api, token, user = _session()
    order = providers.for_user(user)
    spec = {"candidates": candidates, "runs": [{"candidate": name, "tasks": tasks, "trials": 1} for name in candidates],
            "budget_s": workspace.eval_budget_s(), "providers": order}
    color = sys.stdout.isatty()
    several = len(candidates) > 1
    print(f"Evaluating {what}, 1 trial each (GLM-5.3-Flash on {order[0]}).", flush=True)
    try:
        res = launch.run_eval(api, token, user, spec, on_job=lambda job: print(f"HF Job: {getattr(job, 'url', None) or job.id}", flush=True),
                              on_record=lambda r: print(_trial_line(r, color) + (f"   {r['candidate']}" if several else ""), flush=True), package_url=args.package_url)
    except KeyboardInterrupt:
        print("\nStopped: the Job was cancelled and nothing was recorded.")
        return 130
    except launch.EvalJobError as e:
        print(f"The evaluation Job failed: {e}\nNothing was recorded. Run `{again}` again.")
        return 1
    except Exception as e:  # noqa: BLE001 - the Hub refused the Job (credits, token scope) or the network is down
        print(f"The evaluation Job could not run: {type(e).__name__}: {str(e)[:300]}\nNothing was recorded. "
              f"402: the account has no credits left (Settings > Billing on huggingface.co). "
              f"401 or 403: create a token of type \"Write\" at https://huggingface.co/settings/tokens, then run `hf auth login` again.")
        return 1
    scored = [r for r in res.records if r["outcome"] != "infra"]
    lost = len(tasks) * len(candidates) - len(scored)
    lost_note = f" ({lost} trials were lost to infrastructure and do not count)" if lost else ""
    if not scored:
        print(f"No trial was scored{lost_note}. Nothing was recorded: run `{again}` again.")
        return 1
    cost = sum(r.get("cost_usd") or 0.0 for r in res.records) + res.job_usd
    spent = f"{res.seconds:.0f} s, ${cost:.3f}"
    if args.heldout:
        held = workspace.record_heldout(d, {int(name[1:]): [r for r in res.records if r["candidate"] == name] for name in candidates}, trials=1)
        shown = {v["version"]: held.get(v["version"]) or v.get("heldout") for v in wanted}
        print("Held-out: " + ", ".join(f"v{n}: {_pct(h)}" for n, h in shown.items() if h) + f"{lost_note}   {spent}")
        if len(held) < len(candidates):
            print(f"Not every version was scored: run `{again}` again for the missing one.")
            if not held:
                return 1
    else:
        entry = workspace.record(d, version, source, res.records, seconds=round(res.seconds, 1), cost_usd=round(cost, 4), job_url=res.job_url)
        before = f"   (v{versions[-1]['version']}: {workspace.score_text(versions[-1])})" if versions else ""
        print(f"v{version}: {workspace.score_text(entry)}{lost_note}{before}   {spent}")
        print(f"Trajectories: {d / 'traces' / f'v{version}' / 'INDEX.md'}")
    print(f"Chart: {d / 'chart.html'}")
    return 0


def cmd_eval_job(args: argparse.Namespace) -> int:
    from aidev_evolve import evaljob

    return evaljob.cli()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aidev-evolve", description="Score agent.py on Terminal-Bench 2.1 tasks in an HF Job on your own HF account.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="score agent.py as its next version on the 8 validation tasks (one HF Job); run it in the folder that holds agent.py")
    e.add_argument("--dir", default=".", help="your folder (default: the current one)")
    e.add_argument("--heldout", action="store_true", help="score the latest version and version 1 on the 21 held-out tasks instead (one HF Job)")
    e.add_argument("--package-url", default=None, help="another wheel for the Job to install (default: this version's)")
    e.set_defaults(fn=cmd_eval)
    j = sub.add_parser("eval-job", help="(inside the HF Job) run the spec from $AIDEV_EVOLVE_SPEC")
    j.set_defaults(fn=cmd_eval_job)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
