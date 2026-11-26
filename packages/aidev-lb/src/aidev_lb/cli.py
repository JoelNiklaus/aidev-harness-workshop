"""`aidev-lb`: run coding-agent harnesses of your choice on a small Terminal-Bench 2.1 subset (AI Dev NYC workshop).

    aidev-lb check                                  # before the workshop: one model call, one tiny HF Job, one sandbox
    aidev-lb harnesses                              # the 10 harnesses with a speed label
    aidev-lb run --harness pi --harness qwen-code   # one HF Job per harness, 5 tasks each, results in results.json

`run` starts one HF Job per harness in your own namespace. Each Job runs all trials of its harness at once, every
trial in its own HF Sandbox, and `run` shows one line per trial event read from the Job log. Your token pays for
your Jobs, sandboxes and model calls (your HF credits). It is sent only to Hugging Face and is never printed.

`run` also keeps the results in the public bucket <you>/harness-results (created if missing), where a static page can
read them. `--no-bucket` keeps them in the local file only; so does a token that cannot write the bucket. A trial that
ends without a result (an infrastructure error, or the time limit) has none.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aidev_lb import __version__, catalog, check, events, launch, providers, store

WHOAMI = "https://huggingface.co/api/whoami-v2"
# `run` waits until every Job has said that its trials are over (a result or an infrastructure error each). After
# this many seconds from the command's start it stops waiting: cells without a result stay without one and the Jobs
# still running are cancelled.
WAIT_S = 15 * 60
STILL_WAITING_S = 30  # how often `run` says what it still waits for


def _whoami(token: str) -> dict[str, Any]:
    import httpx

    r = httpx.get(WHOAMI, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _job_run(harness: str, tasks: list[str], token: str, user: str, workdir: Path, run_key: str, bucket: str | None) -> int:
    from aidev_lb import job

    return job.run(harness, tasks, token, user, workdir, run_key, bucket=bucket)


@dataclass
class Deps:
    whoami: Callable[[str], dict[str, Any]] = _whoami
    run_job: Callable[..., Any] = launch.submit
    package_status: Callable[[str, str], int] = launch.package_status
    job_logs: Callable[[str, str, str], Iterable[str]] = launch.job_logs
    job_stage: Callable[[str, str, str], str] = check.job_stage
    cancel_job: Callable[[str, str, str], None] = check.cancel_job
    cancel_sandboxes: Callable[[str, str, str], None] = launch.cancel_sandboxes
    job_run: Callable[..., int] = _job_run
    publish: Callable[[Path, str, str], None] = store.publish
    chat: Callable[[str, str, str], str] = check.chat
    sandbox_echo: Callable[[str, str], str] = check.sandbox_echo
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    poll_s: float = 0.5  # how often `run` looks at the clock while no event arrives
    reopen_s: float = 2.0  # pause before a broken log stream is opened again
    check_poll_s: float = 3.0
    check_wait_s: float = 300.0


def _task_count(v: str) -> int:
    try:
        catalog.tasks(int(v))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return int(v)


def _bucket_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bucket", metavar="NAME", default=store.BUCKET, help=f"the public bucket <you>/NAME that also gets the results, for a page that shows "
                   f"them (default {store.BUCKET}, created if missing)")
    p.add_argument("--no-bucket", action="store_true", help="keep the results in the local file only")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aidev-lb", description="Run coding-agent harnesses on a small Terminal-Bench 2.1 subset, on your own HF account.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test this account at home: one model call, one tiny HF Job, one sandbox (under one cent)")
    sub.add_parser("harnesses", help="list the harnesses you can run, with a speed label")
    r = sub.add_parser("run", help="run harnesses: one HF Job per harness, all its trials at once")
    r.add_argument("--harness", action="append", required=True, help="a harness of `aidev-lb harnesses`; repeat for more")
    r.add_argument("--tasks", type=_task_count, default=None, help="how many tasks, from the top of the list (default 5, up to 10)")
    r.add_argument("--out", type=Path, default=store.LOCAL, help="results file (default ./results.json)")
    _bucket_flags(r)
    r.add_argument("--package", help="runner wheel the Jobs install: a URL, or a local wheel file (default: this version's released wheel)")
    j = sub.add_parser("job", help="what a started HF Job runs: every trial of one harness")
    j.add_argument("--harness", required=True)
    j.add_argument("--tasks", required=True, help="task names, comma-separated")
    j.add_argument("--user", required=True)
    return p


def _say(msg: str) -> None:
    print(msg, flush=True)


def _token(environ: Mapping[str, str], token_lookup: Callable[[], str | None] | None) -> str | None:
    if environ.get("HF_TOKEN"):
        return environ["HF_TOKEN"]
    if token_lookup is None:
        from huggingface_hub import get_token

        return get_token()
    return token_lookup()


def _harnesses() -> int:
    hs = catalog.harnesses()
    width = max(len(h["name"]) for h in hs)
    for h in hs:
        _say(f"{h['name']:<{width}}  {h['speed']}" + (f"  ({h['note']})" if h.get("note") else ""))
    _say("\nSpeed is the harness install time in a fresh sandbox (fast: under a minute, slow: 2 to 3 minutes or more).")
    return 0


TOKEN_FIX = "Create a token of type \"Write\" at https://huggingface.co/settings/tokens and run `hf auth login` again"  # the one token fix for attendees


def _publish(deps: Deps, local: Path, bucket: str, token: str, say: Callable[[str], None]) -> bool:
    """Write the local results to the bucket. A failure is one plain line, and the caller goes on with the local file."""
    try:
        deps.publish(local, bucket, token)
    except Exception as e:  # noqa: BLE001
        say(f"Could not write to the bucket {bucket} ({type(e).__name__}: {str(e)[:200]}). Your token probably cannot write. "
            f"The results are kept in this folder only. {TOKEN_FIX}, then run the command again to publish them.")
        return False
    return True


def _unknown(harnesses: list[str]) -> str | None:
    known = [h["name"] for h in catalog.harnesses()]
    bad = [h for h in harnesses if h not in known]
    return f"Unknown harness: {', '.join(bad)}. Choose from: {', '.join(known)}." if bad else None


def _follow(deps: Deps, token: str, user: str, harness: str, job_id: str, q: queue.Queue, stop: threading.Event) -> None:
    """Read one Job's log until the Job is done, and queue each event once. A stream that ends while the Job still
    runs (a network break) is opened again; it replays from the start, so the events already seen are skipped."""
    seen, stage = 0, None
    while not stop.is_set():
        n = 0
        try:
            for text in deps.job_logs(token, user, job_id):
                ev = events.parse(text)
                if ev is None:
                    continue
                n += 1
                if n > seen:
                    seen = n
                    q.put((harness, ev))
            stage = deps.job_stage(token, user, job_id)
        except Exception:  # noqa: BLE001 - a broken stream or status read: try again until the wait is over
            stage = None
        if stage in check.DONE:
            break
        stop.wait(deps.reopen_s)
    q.put((harness, {"event": "stream_closed", "stage": stage}))


def _stop_jobs(deps: Deps, token: str, user: str, job_ids: list[str], run_key: str, say: Callable[[str], None]) -> None:
    """Cancel the Jobs still running and the trial sandboxes they started, so nothing keeps spending credits."""
    try:
        for job_id in job_ids:
            deps.cancel_job(token, user, job_id)
        deps.cancel_sandboxes(token, user, run_key)
    except Exception as e:  # noqa: BLE001
        say(f"could not stop every Job ({type(e).__name__}: {str(e)[:200]}): stop them at https://huggingface.co/settings/jobs")
    else:
        say(f"stopped {len(job_ids)} Job(s) still running, and their sandboxes")


def _run(args: argparse.Namespace, deps: Deps, token: str, user: str) -> int:
    t0 = deps.clock()
    harnesses, tasks = list(dict.fromkeys(args.harness)), catalog.tasks(args.tasks)
    if unknown := _unknown(harnesses):
        print(unknown, file=sys.stderr)
        return 2
    if not args.out.resolve().parent.is_dir():
        print(f"The folder for the results file does not exist: {args.out.resolve().parent}", file=sys.stderr)
        return 2
    store.read_all(args.out)  # a file that is not a results list stops here, before any Job is paid for
    package = args.package or launch.default_package_url()
    if launch.is_url(package):
        try:
            status = deps.package_status(package, token)
        except Exception as e:  # noqa: BLE001
            status = type(e).__name__
        if status != 200:
            print(f"The runner package cannot be downloaded ({status}): {package}. The Jobs would fail right after their start, "
                  f"so nothing was started. Update the tool with the install line of the workshop README, or name a wheel with --package.", file=sys.stderr)
            return 1
    elif not Path(package).is_file():
        print(f"No such wheel file: {package}", file=sys.stderr)
        return 2

    bucket = None if args.no_bucket else f"{user}/{args.bucket}"
    if bucket:
        _say(f"Checking your results bucket {bucket} ...")
    if bucket and not _publish(deps, args.out, bucket, token, _say):  # creates the bucket; asked once, before any Job
        bucket = None  # local only: the Jobs and the later writes leave the bucket alone

    def say(text: str) -> None:
        _say(f"[{deps.clock() - t0:4.0f} s] {text}")

    changed, finished = threading.Event(), threading.Event()

    def publish() -> None:
        try:
            deps.publish(args.out, bucket, token)
        except Exception as e:  # noqa: BLE001 - the local file is complete; the next write brings the bucket up to date
            say(f"could not update the bucket {bucket} ({type(e).__name__}: {str(e)[:200]})")

    def keep_bucket_current() -> None:
        """Its own thread: a bucket write takes seconds, and the lines of the next trials must not wait for it."""
        while not finished.is_set():
            if changed.wait(deps.poll_s):
                changed.clear()
                publish()

    _say(f"aidev-lb {__version__}: {len(harnesses)} harness(es) x {len(tasks)} tasks on {catalog.spec()['model']}, as {user}. "
         f"One HF Job per harness; your HF credits pay.")
    run_key, q, stop = uuid.uuid4().hex[:8], queue.Queue(), threading.Event()
    jobs: dict[str, str] = {}  # harness -> Job id, for the Jobs that started
    page = lambda h: f"https://huggingface.co/jobs/{user}/{jobs[h]}"  # noqa: E731

    def start(harness: str) -> None:
        try:
            job = deps.run_job(token, **launch.job_spec(token, user, harness, tasks, run_key, package=package, bucket=bucket))
        except Exception as e:  # noqa: BLE001
            q.put((harness, {"event": "not_started", "error": f"{type(e).__name__}: {str(e)[:300]}"}))
            return
        jobs[harness] = job.id
        say(f"{harness}  Job page: {page(harness)}")  # at once, before the first log line is read
        _follow(deps, token, user, harness, job.id, q, stop)

    for h in harnesses:
        threading.Thread(target=start, args=(h,), daemon=True, name=f"job-{h}").start()
    publisher = threading.Thread(target=keep_bucket_current, daemon=True, name="bucket")
    if bucket:
        publisher.start()
    following, why = set(harnesses), {}  # why: cell -> why it has no result of its own
    cells = {catalog.cell_id(h, t) for h in harnesses for t in tasks}
    open_cells, queued = set(cells), set(cells)  # trials without an end yet; of those, the ones whose sandbox has not started

    def plural(n: int, word: str) -> str:
        return f"{n} {word}" + ("" if n == 1 else "es" if word.endswith("x") else "s")

    def running() -> list[str]:
        return [jobs[h] for h in following if h in jobs]

    def handle(harness: str, ev: dict[str, Any]) -> None:
        kind = ev["event"]
        if kind == "not_started":
            following.discard(harness)
            open_cells.difference_update(catalog.cell_id(harness, t) for t in tasks)  # nothing to wait for
            say(f"{harness}  Hugging Face did not start the Job ({ev['error']}). Check your credits at https://huggingface.co/settings/billing "
                f"and that the token may start Jobs.")
        elif kind == "stream_closed":
            if ev["stage"] not in check.DONE:
                return  # the log reader stopped because the wait is over; the Job still runs
            if harness in following and ev["stage"] != "COMPLETED":
                say(f"{harness}  Job ended {ev['stage']}: read its log at {page(harness)}")
            following.discard(harness)
            open_cells.difference_update(catalog.cell_id(harness, t) for t in tasks)  # the Job is over
        elif kind == "job_done":
            # The Job's last line. The Job itself needs up to a minute more to end (process exit, stage change, the
            # close of the log stream), and there is nothing to wait for.
            following.discard(harness)
        elif kind == "result":
            store.write_result(ev["result"], local=args.out)  # a Job only reports the cells this command gave it
            changed.set()
            open_cells.discard(ev["result"]["cell_id"])
            say(events.plain(ev))
        else:
            if kind == "infra":
                why[ev["cell_id"]] = "infrastructure error"
                open_cells.discard(ev["cell_id"])
            elif kind in events.PHASE_TEXT:
                queued.discard(ev["cell_id"])
            if (text := events.plain(ev)) is not None:
                say(text)

    def drain() -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                handle(*q.get_nowait())

    try:
        said = t0
        while following and (now := deps.clock()) - t0 < WAIT_S:
            if now - said >= STILL_WAITING_S:
                said = now
                say(f"still waiting: {plural(len(open_cells), 'trial')} ({plural(len(open_cells & queued), 'sandbox')} queued)")
            with contextlib.suppress(queue.Empty):
                handle(*q.get(timeout=deps.poll_s))
        stop.set()
        drain()  # what arrived while the wait ran out still counts
    except KeyboardInterrupt:
        stop.set()
        finished.set()
        drain()
        _stop_jobs(deps, token, user, running(), run_key, say)
        if bucket:
            publisher.join()
            publish()
        say(f"interrupted: the results so far are in {args.out}" + (f" and at {store.public_url(bucket)}" if bucket else ""))
        return 130
    if following:
        _stop_jobs(deps, token, user, running(), run_key, say)
    results = store.read_all(args.out)
    have = {r["cell_id"] for r in results}
    for h in harnesses:
        for t in tasks:
            if (cid := catalog.cell_id(h, t)) not in have:
                reason = why.get(cid) or (f"no result after {WAIT_S} s" if h in following else "the Job ended without one")
                say(f"{h}  {t}  no result: {reason}")
    finished.set()
    if bucket:
        publisher.join()
        publish()
    for h in harnesses:
        mine = [r for r in results if r["harness"] == h and r["cell_id"] in cells]
        missing = f", {len(tasks) - len(mine)} of {len(tasks)} tasks without a result" if len(mine) < len(tasks) else ""
        say(f"{h}  summary: {sum(r['passed'] for r in mine)} of {len(mine)} passed{missing}, ${sum(r['cost_usd'] for r in mine):.4f}")
    _say(f"Results: {len(results)} records in {args.out}" + (f" and at {store.public_url(bucket)}" if bucket else ""))
    return 0 if jobs else 1


def _check(deps: Deps, token: str, user: str) -> int:
    """The setup's self-test: every problem in one run, each with its fix."""
    model = catalog.spec()["model"]
    problems = []
    order = providers.order(user, providers.load())
    for i, provider in enumerate(order):
        try:
            reply = deps.chat(token, model, provider)
            _say(f"model call ok: {model} on {provider} answered {reply[:40]!r}")
            break
        except check.ProviderBusy as e:
            if i + 1 < len(order):
                _say(f"{provider} did not serve the call ({str(e)[:120]}); trying {order[i + 1]}")
                continue
            problems.append(f"No workshop provider served the model call ({', '.join(order)}; last: {str(e)[:200]}). If the answer is 401 or 403: "
                            f"{TOKEN_FIX}. Otherwise run the check again in a few minutes.")
        except Exception as e:  # noqa: BLE001
            problems.append(f"The model call failed ({type(e).__name__}: {str(e)[:200]}). Your account needs credits and a token "
                            f"that can write. {TOKEN_FIX}.")
            break
    try:
        job = deps.run_job(token, **check.job_spec(user))
    except Exception as e:  # noqa: BLE001
        problems.append(f"Hugging Face did not start the test Job ({type(e).__name__}: {str(e)[:200]}). Your token needs "
                        f"the permission \"Start and manage Jobs\", and your account needs credits.")
    else:
        url = f"https://huggingface.co/jobs/{user}/{job.id}"
        _say(f"test Job started: {url}")
        waited = 0.0
        try:
            stage = deps.job_stage(token, user, job.id)
            while stage not in check.DONE and waited < deps.check_wait_s:
                deps.sleep(deps.check_poll_s)
                waited += deps.check_poll_s
                stage = deps.job_stage(token, user, job.id)
        except Exception as e:  # noqa: BLE001
            stage = None
            problems.append(f"Could not read the status of the test Job {job.id} ({type(e).__name__}: {str(e)[:200]}). Look at {url}; "
                            f"if it says COMPLETED, you are fine.")
        if stage is None:
            pass  # already reported
        elif stage == "COMPLETED":
            _say(f"test Job ok: COMPLETED after about {waited:.0f} s")
        elif stage in check.DONE:
            problems.append(f"The test Job {job.id} ended {stage}: read its log at {url}")
        else:
            with contextlib.suppress(Exception):
                deps.cancel_job(token, user, job.id)
            problems.append(f"The test Job {job.id} did not finish within {deps.check_wait_s:.0f} s (still {stage}), so it was "
                            f"cancelled. Run the check again; if it repeats, read the Job's log at {url}.")
    try:
        _say(f"sandbox ok: {deps.sandbox_echo(token, user)[:40]!r}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"The HF Sandbox did not start or run ({type(e).__name__}: {str(e)[:200]}). Your token needs the "
                        f"permission \"Start and manage Jobs\" (sandboxes are Jobs), and your account needs credits.")
    if problems:
        for msg in problems:
            print(msg, file=sys.stderr)
        return 1
    _say(f"ready: {user} can call the workshop model, run HF Jobs and use HF Sandboxes. See you at the workshop.")
    return 0


def main(argv: list[str] | None = None, deps: Deps | None = None, environ: Mapping[str, str] | None = None,
         token_lookup: Callable[[], str | None] | None = None) -> int:
    args = _parser().parse_args(argv)
    deps = deps or Deps()
    environ = os.environ if environ is None else environ
    if args.cmd == "harnesses":
        return _harnesses()
    try:
        return _command(args, deps, environ, token_lookup)
    except store.BadResultsFile as e:
        print(str(e), file=sys.stderr)
        return 2


def _command(args: argparse.Namespace, deps: Deps, environ: Mapping[str, str], token_lookup: Callable[[], str | None] | None) -> int:
    token = _token(environ, token_lookup)
    if not token:
        print("No Hugging Face token found. Run `hf auth login` with a token of type \"Write\", or set HF_TOKEN.", file=sys.stderr)
        return 2
    if args.cmd == "job":  # inside the HF Job: the token is the Job secret, the user comes from the laptop
        workdir = Path(environ.get("AIDEV_LB_WORKDIR") or Path.home() / ".cache" / "aidev-lb" / "jobs")
        return deps.job_run(args.harness, args.tasks.split(","), token, args.user, workdir, environ.get("AIDEV_LB_RUN", ""),
                            environ.get("AIDEV_LB_BUCKET") or None)
    try:
        me = deps.whoami(token)
    except Exception as e:  # noqa: BLE001
        print(f"Hugging Face did not accept the token ({type(e).__name__}). Check it with `hf auth whoami` or create a new one.", file=sys.stderr)
        return 2
    user = me.get("name")
    if not me.get("canPay"):
        print(f"The account {user} has no credits to pay for Jobs, sandboxes and inference. Redeem the workshop coupon at https://huggingface.co/settings/billing.", file=sys.stderr)
        return 2
    if args.cmd == "check":
        return _check(deps, token, user)
    return _run(args, deps, token, user)
