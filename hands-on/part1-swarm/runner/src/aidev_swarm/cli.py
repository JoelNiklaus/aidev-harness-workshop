"""`aidev-swarm`: evaluate coding-agent harnesses together on Terminal-Bench 2.1 (AI Dev NYC workshop).

    aidev-swarm check               # before the workshop: one model call and one tiny HF Job, nothing claimed
    aidev-swarm launch [--n K] [--parallel N]   # submit an HF Job that runs your trials on HF (8 at once); the laptop can close
    aidev-swarm run [--loop | --n K] [--parallel N] [--dry]
    aidev-swarm run --local --harness opencode --task nginx-request-logging    # one trial, no coordinator

`run` claims a cell of the workshop grid, runs it with Harbor on an HF Sandbox and submits the result
(it is what the launched Job runs; it needs the `run` extra). Your token pays for your own Job, sandbox
and model calls (your HF credits). It is sent only to Hugging Face and, as the protocol's Bearer header,
to the workshop coordinator; it is never printed.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import random
import signal
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from aidev_swarm import __version__, check, launch, providers

WHOAMI = "https://huggingface.co/api/whoami-v2"
DEFAULT_COORDINATOR = "https://agent-harness-workshop-aidev-swarm.hf.space"  # Space agent-harness-workshop/aidev-swarm (coordinator deploy.DEFAULT_SPACE)
MAX_FAILURES_IN_A_ROW = 3
# The coordinator's lease is 90 s (store.LEASE_S): a beat every 30 s survives two lost beats, and a dead runner's
# cell returns ~90 s after its last beat. 500 runners at one beat per 24-30 s send about 19 requests/s, below the
# ~30/s from one IP that the HF edge took without a 429 (coordinator/PROGRESS.md).
HEARTBEAT_S = 30
# A trial whose provider answered this many calls with 429/5xx (or that ended before any model call) moves the
# attendee's next trials to the next provider of their list (providers.order; PROTOCOL.md "Providers").
PROVIDER_ERRORS_TO_SWITCH = 3
# Trials one run may have going at once (--parallel), one lease each; the coordinator allows one lease more
# (swarm_coordinator.store.MAX_LEASES = 9) for a slot that skips a failed cell.
MAX_PARALLEL = 8
# A slot at the attendee's share of the room (the coordinator's 409 with `share`) asks again after this, doubling up to
# SLOT_WAIT_MAX_S while it keeps being refused (also when every free cell is taken): 500 Jobs x 8 slots from a few HF
# egress IPs would otherwise send ~130 claims/s at the HF edge's per-IP limit.
SHARE_WAIT_S = 30
SLOT_WAIT_MAX_S = 300


def slot_wait(base_s: float, refusals: int) -> float:
    return min(SLOT_WAIT_MAX_S, base_s * 2 ** refusals)


def data_text(name: str) -> str:
    return resources.files("aidev_swarm").joinpath("data", name).read_text()


def _whoami(token: str) -> dict[str, Any]:
    import httpx

    r = httpx.get(WHOAMI, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _coordinator(url: str, token: str):
    from aidev_swarm.coordinator import Coordinator

    return Coordinator(url, token)


def board_url(flag: str | None, environ: Mapping[str, str]) -> str:
    """The board both hands-ons post to: --board, else $AIDEV_BOARD (shared with aidev-evolve; a staging Space for a dry
    run), else $AIDEV_SWARM_URL (what `launch` sets inside the Job), else the workshop Space."""
    return flag or environ.get("AIDEV_BOARD") or environ.get("AIDEV_SWARM_URL") or DEFAULT_COORDINATOR


def _workdir(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("AIDEV_SWARM_WORKDIR") or Path.home() / ".cache" / "aidev-swarm" / "jobs")


def _run_trial(claim, grid, token, user, workdir):
    from aidev_swarm.trial import run_trial

    return run_trial(claim, grid, token, user, workdir)


def _make_sandbox_id(token: str) -> Callable[[str, str], str | None]:
    def lookup(cell_id: str, user: str) -> str | None:
        from huggingface_hub import HfApi

        from aidev_swarm.trial import sandbox_id

        return sandbox_id(HfApi(token=token), cell_id, user)

    return lookup


@dataclass
class Deps:
    whoami: Callable[[str], dict[str, Any]] = _whoami
    coordinator: Callable[[str, str], Any] = _coordinator
    run_trial: Callable[..., Path] = _run_trial
    sandbox_id: Callable[[str, str], str | None] | None = None
    run_job: Callable[..., Any] = launch.submit
    package_status: Callable[[str], int] = launch.package_status
    chat: Callable[[str, str, str], str] = check.chat
    job_stage: Callable[[str, str, str], str] = check.job_stage
    sandbox_echo: Callable[[str, str], str] = check.sandbox_echo
    cancel_job: Callable[[str, str, str], None] = check.cancel_job
    sleep: Callable[[float], None] = time.sleep
    heartbeat_s: float = HEARTBEAT_S
    check_poll_s: float = 3.0
    check_wait_s: float = 300.0


def _positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def _parallel(v: str) -> int:
    n = int(v)
    if not 1 <= n <= MAX_PARALLEL:
        raise argparse.ArgumentTypeError(f"must be from 1 to {MAX_PARALLEL}")
    return n


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aidev-swarm", description="Swarm-evaluate coding-agent harnesses on Terminal-Bench 2.1 (AI Dev NYC workshop).")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test this account at home: one model call and one tiny HF Job (a few cents at most), nothing claimed")
    la = sub.add_parser("launch", help="submit an HF Job that claims and runs trials on HF (your laptop can close)")
    la.add_argument("--n", type=_positive, help="run K trials, then stop (default: keep going until the grid is full)")
    la.add_argument("--board", "--coordinator", dest="coordinator", help=f"board (coordinator) URL (default: $AIDEV_BOARD or {DEFAULT_COORDINATOR})")
    la.add_argument("--package", help="runner wheel URL the Job installs (default: this version's wheel)")
    la.add_argument("--parallel", type=_parallel, default=launch.DEFAULT_PARALLEL,
                    help=f"trials the Job runs at once, each in its own HF Sandbox (default {launch.DEFAULT_PARALLEL}; a full room gets fewer)")
    r = sub.add_parser("run", help="claim a cell, run it, submit the result (what the launched Job runs)")
    r.add_argument("--board", "--coordinator", dest="coordinator", help=f"board (coordinator) URL (default: $AIDEV_BOARD or {DEFAULT_COORDINATOR})")
    r.add_argument("--dry", action="store_true", help="check token and credits, claim nothing")
    mode = r.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="claim the next cell after each trial until the grid is full")
    mode.add_argument("--n", type=_positive, help="run K trials, then stop")
    r.add_argument("--parallel", type=_parallel, default=1, help="trials to run at once (default 1)")
    r.add_argument("--local", action="store_true", help="run one trial without a coordinator (needs --harness and --task)")
    r.add_argument("--harness")
    r.add_argument("--task")
    r.add_argument("--trial", type=int, default=0)
    return p


def _say(msg: str) -> None:
    print(f"[aidev-swarm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def beat_wait(every_s: float, rng: random.Random) -> float:
    """80-100% of the interval at random, so runners launched in the same minute do not beat in step."""
    return every_s * rng.uniform(0.8, 1.0)


def _heartbeats(coord: Any, cid: str, every_s: float, stop: threading.Event, started: float) -> None:
    """Runs in its own thread, so the lease stays renewed whatever the trial does."""
    from aidev_swarm.coordinator import LeaseLost

    rng, n = random.Random(), 0
    while not stop.wait(beat_wait(every_s, rng)):
        try:
            coord.heartbeat(cid)
            n += 1
            if n % 10 == 0:  # about every 5 min
                _say(f"still running ({(time.monotonic() - started) / 60:.0f} min)")
        except LeaseLost as e:
            _say(f"lease lost on {cid} ({str(e)[:120]}): the cell expired and went to another runner; "
                 f"this trial's result counts only if that runner gives it back")
        except Exception as e:  # noqa: BLE001 - a lost heartbeat must not kill the trial; the next beat tries again
            _say(f"heartbeat failed: {type(e).__name__}: {str(e)[:120]}")


def _infra(result: dict[str, Any]) -> bool:
    """Ended before the agent made any model call: the cell goes back to the queue instead of scoring a fail."""
    return bool(result["error"]) and not result["n_calls"]


def cell_id(harness: str, task: str, trial: int) -> str:
    """The coordinator's cell id format (coordinator/src/swarm_coordinator/grid.py)."""
    return f"{harness}__{task}__{trial}"


@contextlib.contextmanager
def _sigterm_as_interrupt() -> Iterator[None]:
    """A cancelled or timed-out HF Job gets SIGTERM: treat it like Ctrl-C, so the cell is released. Only the first
    signal interrupts; later ones are ignored so they cannot cut the release short (SIGKILL follows anyway)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    before = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, before)


@dataclass
class Session:
    """What every trial of one `run` shares."""

    deps: Deps
    coord: Any  # None in --local mode
    grid: dict[str, Any]
    prices: dict[str, Any]
    token: str
    user: str
    workdir: Path
    board: str = ""
    held: set[str] = field(default_factory=set)  # cells this run holds a lease on: released on every way out
    providers: list[str] = field(default_factory=list)  # this attendee's providers, the current one first
    parallel: int = 1  # slots: trials running at once, each on its own lease
    # The leases the coordinator lets this attendee hold now (its `share`, from every claim answer): slots below it claim,
    # the slot at it probes (backing off while refused), slots above it wait without asking. It starts at the
    # coordinator's slow start (2) and drops when the room fills up, so a full room's Jobs leave their extra slots idle.
    share: int = 2
    full: threading.Event = field(default_factory=threading.Event)  # a slot found the grid full: every slot stops
    lock: threading.Lock = field(default_factory=threading.Lock)  # held cells and kept results, shared by the slots

    def hold(self, cid: str) -> None:
        with self.lock:
            self.held.add(cid)

    def drop(self, cid: str) -> None:
        with self.lock:
            self.held.discard(cid)

    @property
    def provider(self) -> str | None:
        return self.providers[0] if self.providers else None

    def next_provider(self, why: str) -> None:
        """Move the current provider to the end of the list: the next trials go to the next one."""
        if len(self.providers) > 1:
            self.providers.append(self.providers.pop(0))
            _say(f"{why}: switching to {self.providers[0]} for the next trials")

    def release(self, cid: str, why: str, *, quick: bool = False) -> None:
        """Give the cell back; when the coordinator cannot be reached, its lease expires instead (90 s)."""
        self.drop(cid)
        if self.coord is None:
            _say(why)
            return
        try:
            self.coord.release(cid, quick=quick)
            _say(f"{why}: cell released")
        except Exception as e:  # noqa: BLE001 - the original failure is the news; never mask it
            _say(f"{why}; releasing the cell failed too ({type(e).__name__}), so its lease expires on its own")

    def release_held(self, why: str) -> None:
        """On the way out (exception, SIGTERM, interpreter exit): give back every cell still held, fast."""
        with self.lock:
            held = sorted(self.held)
        for cid in held:
            self.release(cid, why, quick=True)

    # Results the coordinator did not take (HF edge rate limit, outage) wait here for the next try.
    @property
    def unsent(self) -> Path:
        return self.workdir / "unsent"

    def keep_unsent(self, cid: str, result: dict[str, Any]) -> Path:
        self.unsent.mkdir(parents=True, exist_ok=True)
        path = self.unsent / f"{urllib.parse.quote(cid, safe='')}.json"
        path.write_text(json.dumps(result))
        return path

    def resend_unsent(self) -> None:
        with self.lock:  # one slot at a time, so a kept result is sent once
            self._resend_unsent()

    def _resend_unsent(self) -> None:
        from aidev_swarm.coordinator import LeaseLost

        for path in sorted(self.unsent.glob("*.json")) if self.unsent.is_dir() else []:
            result = json.loads(path.read_text())
            try:
                self.coord.submit(result["cell_id"], result)
            except LeaseLost as e:
                path.unlink()
                _say(f"dropped the kept result of {result['cell_id']}: the coordinator gave the cell to someone else ({str(e)[:120]})")
            except Exception as e:  # noqa: BLE001 - keep it for the next try
                _say(f"kept result of {result['cell_id']} still not accepted ({type(e).__name__}); will try again")
            else:
                path.unlink()
                _say(f"submitted the kept result of {result['cell_id']}")


def _one(claim: dict[str, Any], s: Session, *, hold_on_failure: bool = False) -> tuple[int, str | None]:
    """Run one claimed cell. Returns (exit code, the failed cell still held for the caller to release).

    With `hold_on_failure`, a setup failure keeps the lease so the caller can claim the next cell first
    (the coordinator returns a released cell to the head of its queue)."""
    from aidev_swarm.result import build_result

    cid = claim["cell_id"]
    verb = "claimed" if s.coord is not None else "running"
    forced = (claim.get("run_config") or {}).get("provider")  # the coordinator can pin a provider for everyone
    provider = forced or s.provider
    _say(f"{verb} {claim['harness']} on {claim['task']} (trial {claim['trial']}), model {claim['model']} on {provider}: starting an HF Sandbox")
    stop, started = threading.Event(), time.monotonic()
    if s.coord is not None:
        threading.Thread(target=_heartbeats, args=(s.coord, cid, s.deps.heartbeat_s, stop, started), daemon=True).start()

    def failed(why: str) -> tuple[int, str | None]:
        if hold_on_failure and s.coord is not None:
            _say(why)
            return 1, cid
        s.release(cid, why)
        return 1, None

    try:
        trial_dir = s.deps.run_trial(claim, {**s.grid, "provider": provider}, s.token, s.user, s.workdir)
        lookup = s.deps.sandbox_id or _make_sandbox_id(s.token)
        result = build_result(claim, trial_dir, s.prices[claim["model"]], hf_user=s.user, runner_version=__version__, provider=provider,
                              sandbox_id=lookup(cid, s.user))
    except KeyboardInterrupt:
        s.release(cid, "interrupted", quick=True)
        return 130, None
    except Exception as e:  # noqa: BLE001
        return failed(f"trial crashed ({type(e).__name__}: {str(e)[:200]})")
    finally:
        stop.set()
    if not forced and (_infra(result) or result["n_provider_errors"] >= PROVIDER_ERRORS_TO_SWITCH):
        s.next_provider(f"{provider} answered {result['n_provider_errors']} calls with 429/5xx" if result["n_provider_errors"]
                        else f"the trial on {provider} ended before any model call")
    if s.coord is None:
        print(json.dumps(result, indent=2))
        return 0, None
    if _infra(result):
        return failed(f"infrastructure failure before any model call ({result['error'][:200]})")
    try:
        s.coord.submit(cid, result)
        s.drop(cid)
    except Exception as e:  # noqa: BLE001 - keep the finished result: on screen and on disk for the next try
        from aidev_swarm.coordinator import LeaseLost

        s.drop(cid)  # no release: the kept result is sent again, and the stopped heartbeat lets the lease run out
        print(json.dumps(result, indent=2))
        kept = "" if isinstance(e, LeaseLost) else f"; kept in {s.keep_unsent(cid, result)} and sent again before the next claim"
        _say(f"result not accepted by the coordinator ({type(e).__name__}: {str(e)[:200]}); it is printed above{kept}")
        return 1, None
    verdict = "PASS" if result["passed"] else "FAIL"
    _say(f"{verdict} {claim['harness']} {claim['task']}: {result['n_calls']} model calls, {result['n_input_tokens'] + result['n_output_tokens']:,} tokens, "
         f"${result['cost_usd']:.4f}, {result['wall_s'] or 0:.0f} s. Submitted. Board: {s.board}")
    return 0, None


def _claim_next(s: Session, held: str | None) -> dict[str, Any] | None:
    """The next cell for one slot. The run holds at most one lease per slot (`limit` = s.parallel); one lease more is
    allowed for a cell still `held` after its setup failure, which is released only after the next claim, so it is
    not handed straight back, and for a lease another runner of the same attendee holds (a second Job)."""
    from aidev_swarm.coordinator import AtLimit

    def got(c: dict[str, Any] | None) -> dict[str, Any] | None:
        if c is not None:
            s.hold(c["cell_id"])
            if isinstance(c.get("share"), int):
                s.share = c["share"]
        return c

    spare = s.parallel + 1
    if held is not None:
        try:
            c = s.coord.claim(spare)
        except AtLimit:  # another runner of this user holds the spare lease: give the cell back, then claim normally
            s.release(held, "skipped after its setup failure")
        else:
            s.release(held, "skipped after its setup failure")
            return got(c)
    try:
        return got(s.coord.claim(s.parallel))
    except AtLimit:
        return got(s.coord.claim(spare))


def _run_cells(s: Session, url: str, limit: int | None) -> int:
    """Claim and run cells until `limit` trials ran (None: until the grid is full). In a multi-trial run, a setup
    failure skips to the next cell; MAX_FAILURES_IN_A_ROW failed trials in a row stop the run. Whatever ends the
    run (an exception, SIGTERM from a cancelled Job), the cells it still holds are given back on the way out."""
    try:
        if s.parallel == 1:
            return _claim_and_run(s, url, limit)
        return _slots(s, url, limit)
    except KeyboardInterrupt:  # Ctrl-C or SIGTERM outside a trial (a trial handles its own)
        _say("interrupted")
        return 130
    finally:
        s.release_held("stopping")


def _slots(s: Session, url: str, limit: int | None) -> int:
    """s.parallel slots, each a thread that claims and runs cells on its own lease (with its own heartbeats); `limit`
    trials in all, split over the slots. Signals reach the main thread, which waits here: on Ctrl-C or SIGTERM it
    returns and the caller releases every cell still held (the slots' sandboxes end with the Job)."""
    n = s.parallel if limit is None else min(s.parallel, limit)
    shares = [None] * n if limit is None else [limit // n + (i < limit % n) for i in range(n)]
    rcs: list[int] = [0] * n

    def slot(i: int) -> None:
        try:
            rcs[i] = _claim_and_run(s, url, shares[i], slot=i)
        except Exception as e:  # noqa: BLE001 - one slot's crash must not take the others down
            _say(f"slot {i + 1} stopped: {type(e).__name__}: {str(e)[:200]}")
            rcs[i] = 1

    threads = [threading.Thread(target=slot, args=(i,), daemon=True, name=f"slot-{i + 1}") for i in range(n)]
    _say(f"running up to {n} trials at once")
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(0.5)  # a bounded join keeps the main thread able to take KeyboardInterrupt
    return max(rcs)


def _claim_and_run(s: Session, url: str, limit: int | None, slot: int = 0) -> int:
    from aidev_swarm.coordinator import AllTaken, AtLimit

    held: str | None = None
    ran = in_a_row = rc = refusals = 0
    waiting = False
    while limit is None or ran < limit:
        if s.parallel > 1 and s.full.is_set():
            if held:
                s.release(held, "skipped after its setup failure")
            return rc
        if s.parallel > 1 and slot > s.share and held is None:  # beyond the share: only the slot at the share asks the coordinator
            s.deps.sleep(SHARE_WAIT_S)
            continue
        s.resend_unsent()
        try:
            claim = _claim_next(s, held)
        except AllTaken as e:
            if held:
                s.release(held, "skipped after its setup failure")
                held = None
            if not waiting:
                _say(f"every free cell is taken; waiting in case a running trial's cell comes back (asking every {e.retry_after:.0f} s"
                     + (", less often while it stays so)" if s.parallel > 1 else ")"))
                waiting = True
            s.deps.sleep(slot_wait(e.retry_after, refusals) if s.parallel > 1 else e.retry_after)
            refusals += 1
            continue
        except AtLimit as e:
            if s.parallel > 1:  # the room is full: this attendee's share is below its slot count for now
                if e.share is not None:
                    s.share = e.share
                if not waiting:
                    _say(f"at your share of the room ({e.share or len(e.held)} cells at once for now); asking again in {SHARE_WAIT_S} s, "
                         f"then less often while it stays so")
                    waiting = True
                s.deps.sleep(slot_wait(SHARE_WAIT_S, refusals))
                refusals += 1
                continue
            print(f"You already hold {len(e.held)} cells ({', '.join(e.held)}): other runs of yours are working on them. "
                  f"Wait for them (or stop them with `hf jobs cancel`); a lease expires 90 s after its last heartbeat.", file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"The coordinator at {url} did not hand out a cell ({type(e).__name__}: {str(e)[:200]}). Try again in a minute.", file=sys.stderr)
            return 1
        held, waiting, refusals = None, False, 0
        if claim is not None and s.parallel > 1 and not isinstance(claim.get("share"), int):
            s.share = max(s.share, slot + 1)  # a coordinator that names no share: this slot's claim was allowed
        if claim is None:
            s.full.set()
            _say(f"nothing left to claim: the grid is full. Board: {url}")
            return 0
        ran += 1
        more = limit is None or ran < limit
        rc, held = _one(claim, s, hold_on_failure=more)
        if rc == 130:
            return rc
        in_a_row = 0 if rc == 0 else in_a_row + 1
        if in_a_row >= MAX_FAILURES_IN_A_ROW:
            if held:
                s.release(held, "stopping")
            _say(f"stopping: {in_a_row} trials in a row failed. Read the errors above; the board is at {url}")
            return 1
    return rc


def _token(environ: Mapping[str, str], token_lookup: Callable[[], str | None] | None) -> str | None:
    if environ.get("HF_TOKEN"):
        return environ["HF_TOKEN"]
    if token_lookup is None:
        from huggingface_hub import get_token

        return get_token()
    return token_lookup()


def _launch(args: argparse.Namespace, deps: Deps, token: str, user: str, url: str) -> int:
    package = args.package or launch.default_package_url()
    try:
        status = deps.package_status(package)
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}"
    if status != 200:
        print(f"The runner package cannot be downloaded ({status}): {package}. The Job would fail right after its start, "
              f"so nothing was launched. Ask the workshop team for the current package URL.", file=sys.stderr)
        return 1
    spec = launch.job_spec(token, user, url, n=args.n, package_url=package, parallel=args.parallel)
    try:
        job = deps.run_job(token, **spec)
    except Exception as e:  # noqa: BLE001
        print(f"Hugging Face did not start the Job ({type(e).__name__}: {str(e)[:300]}). "
              f"Check your credits at https://huggingface.co/settings/billing and that the token may start Jobs.", file=sys.stderr)
        return 1
    what = (f"{args.n} trial(s)" if args.n else "trials until the grid is full") + f", up to {min(args.parallel, args.n or args.parallel)} at once"
    _say(f"launched HF Job {job.id} in {user}'s namespace: it runs {what} on Hugging Face. You can close the laptop.")
    print(f"  Job page: https://huggingface.co/jobs/{user}/{job.id}")
    print(f"  Logs:     uvx hf jobs logs -f {job.id}")
    print(f"  Stop:     uvx hf jobs cancel {job.id}")
    print(f"  Board:    {url}")
    return 0


def _check(deps: Deps, token: str, user: str) -> int:
    """The prep email's self-test: every problem in one run, each with its fix."""
    grid = json.loads(data_text("grid.json"))  # the same packaged grid `run` claims against
    problems = []
    order = providers.order(user, providers.load())
    for i, provider in enumerate(order):
        try:
            reply = deps.chat(token, grid["model"], provider)
            _say(f"model call ok: {grid['model']} on {provider} answered {reply[:40]!r}")
            break
        except check.ProviderBusy as e:
            if i + 1 < len(order):
                _say(f"{provider} busy ({str(e)[:120]}); trying {order[i + 1]}")
                continue
            problems.append(f"Every workshop provider was busy ({', '.join(order)}; last: {str(e)[:200]}). Run the check again in a few minutes.")
        except Exception as e:  # noqa: BLE001
            problems.append(f"The model call failed ({type(e).__name__}: {str(e)[:200]}). Your token needs the permission "
                            f"\"Make calls to Inference Providers\", and your account needs credits.")
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
                            f"cancelled. Run the check again; if it repeats, reply to the workshop email with this line.")
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
    token = _token(environ, token_lookup)
    if not token:
        print("No Hugging Face token found. Run `hf auth login` (token with Inference Providers and Jobs permissions) or set HF_TOKEN.", file=sys.stderr)
        return 2
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
    url = board_url(args.coordinator, environ)
    if args.cmd == "launch":
        return _launch(args, deps, token, user, url)
    grid, prices = json.loads(data_text("grid.json")), json.loads(data_text("prices.json"))
    order = providers.order(user, providers.load())
    if args.local:
        if not (args.harness and args.task):
            print("--local needs --harness and --task", file=sys.stderr)
            return 2
        claim = {"cell_id": cell_id(args.harness, args.task, args.trial), "harness": args.harness, "task": args.task, "trial": args.trial, "model": grid["model"], "run_config": {}}
        with _sigterm_as_interrupt():
            return _one(claim, Session(deps, None, grid, prices, token, user, _workdir(environ), providers=order))[0]
    if args.dry:
        _say(f"ready: HF user {user} can pay; would claim from {url} (model {grid['model']}, {len(grid['harnesses'])} harnesses, {len(grid['tasks'])} tasks). Nothing claimed.")
        return 0
    session = Session(deps, deps.coordinator(url, token), grid, prices, token, user, _workdir(environ), board=url, providers=order,
                      parallel=args.parallel)
    atexit.register(session.release_held, "exiting")  # backstop for an exit that bypasses the finally in _run_cells
    try:
        with _sigterm_as_interrupt():
            return _run_cells(session, url, None if args.loop else (args.n or 1))
    finally:
        atexit.unregister(session.release_held)
