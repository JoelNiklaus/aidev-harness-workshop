# aidev-swarm (runner)

The attendee tool of the Part 1 swarm (see `../PROTOCOL.md`). `aidev-swarm launch` starts **one HF Job in
your own namespace**. The Job claims cells of the workshop grid (harness x Terminal-Bench 2.1 task x trial)
from the coordinator, runs each with Harbor on an HF Sandbox, and submits the results to the live board.
Everything runs on Hugging Face: your laptop only submits the Job (about 1 s) and can close afterwards.
Your HF credits pay for your Job (cpu-basic, $0.01/h), the sandboxes and the GLM-5.3-Flash calls.

## Attendee: run it

```bash
# once, before the workshop: uv installed, HF account with credits, and a fine-grained token with the
# permissions "Make calls to Inference Providers" and "Start and manage Jobs" saved with
uvx hf auth login
uvx --from "<WHEEL_URL>" aidev-swarm check           # one model call, one tiny Job, one sandbox command (~15 s, < $0.01)

# in the room
uvx --from "<WHEEL_URL>" aidev-swarm launch          # trials until the grid is full (Job timeout 3 h)
uvx --from "<WHEEL_URL>" aidev-swarm launch --n 2    # or exactly 2 trials
```

`launch` prints the Job page, `hf jobs logs <id>` (follow it with `-f`), `hf jobs cancel <id>`, and the board
URL. Or paste `AGENT_PROMPT.md` into your coding agent and let it do this. In the room, the coordinator's
`/join` page (the slide's QR code) shows both with Copy buttons.

`<WHEEL_URL>` is today `https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/swarm-v0.4.3/aidev_swarm-0.4.3-py3-none-any.whl`
(a release asset of this public repo: no login needed; the Job downloads the same file). The laptop install is
26 MB (httpx and huggingface_hub; 4 s with an empty uv cache). Harbor and the harness wiring (about 290 MB)
install only inside the Job (the `run` extra, about 8 s there).

Other commands:

- `aidev-swarm check`: the prep email's self-test (token, credits, GLM-5.3-Flash call, tiny Job, sandbox), claims nothing.
- `aidev-swarm run --dry`: check token and credits only, claim nothing (works with the light install).
- `aidev-swarm run [--loop | --n K]`: what the Job runs (needs `aidev-swarm[run]`).
- `aidev-swarm run --local --harness pi --task nginx-request-logging`: one trial, no coordinator (take-home).

The coordinator defaults to `https://agent-harness-workshop-aidev-swarm.hf.space` (the board). Override it with `--board`
or `export AIDEV_BOARD=<url>`, the same setting `aidev-evolve` reads (a dry run points both at the staging Space).

## How it works

- `launch.py` builds the Job: image `ghcr.io/astral-sh/uv:python3.12-bookworm`, cpu-basic, the token only as
  the Job secret `HF_TOKEN`; the command downloads this version's wheel from the GitHub release (no
  credentials sent) and runs
  `uvx --from "aidev-swarm[run] @ file://<wheel>" aidev-swarm run --loop | --n K`.
- `trial.py` reuses the Harness Leaderboard's wiring (`hlb` 0.1.11, pinned to the deployed Space revision; it stops
  an agent whose time is up before the tests start):
  per-harness Harbor configs, the HF Sandbox environment and the in-sandbox meter. Harnesses get a dummy
  key; only the meter inside the sandbox holds your token. Sandboxes run in your namespace.
- `result.py` builds the protocol's Result from the trial folder: tokens from the meter log, cost =
  tokens x `data/prices.json` (cached input at its rate).
- `coordinator.py` speaks the protocol (claim, heartbeat, submit, release; User-Agent `aidev-swarm/<version>`,
  the coordinator refuses claims from runners older than 0.4.3). Transient errors (429 from the HF edge's
  per-IP limit, 5xx, network) are retried with capped backoff: submit for 35 min, claim 10 min, heartbeat
  25 s (the next beat retries), release 2 min (6 s on the way out of a cancelled Job). A result the
  coordinator still does not take is kept in `<workdir>/unsent/` and sent again before the next claim.
- Leases are short: a background thread heartbeats every 24-30 s (random, so runners do not beat in step)
  and the coordinator frees a cell 90 s after its last heartbeat, so the cell of a Job that dies without a
  word (SIGKILL, lost node) returns to the queue within about 1.5 min. A slow trial keeps its cell as long
  as it heartbeats, however long it runs. When every free cell is taken (204 with Retry-After), a looping
  runner waits and asks again, so it picks up cells that come back; a plain 204 means the grid is full.
- In a multi-trial run, a trial that fails before any model call (setup, sandbox start) keeps its lease,
  the next cell is claimed (`?n=2`) and only then the failed cell is released, so the queue does not hand
  it straight back. Three failed trials in a row stop the run. Every way out the runner can catch gives
  its cells back at once: a cancelled or timed-out Job (SIGTERM), Ctrl-C, crashes, and an exit hook.
- A user can hold two leases: a second runner of the same attendee claims with `?n=2` after a 409.
- `providers.py` spreads the room over GLM-5.3-Flash providers: each attendee gets one provider from
  `data/providers.json` (novita 3 : zai-org 1, the providers the Harness Leaderboard admits for the model),
  picked by a weighted hash of the HF username, so the same attendee always gets the same one. Inside a trial
  the meter sends a call novita answers with 429 on to zai-org; a trial with 3+ calls answered 429/5xx, or one
  that ended before any model call, moves the next trials to the next provider. Every result records
  `provider` and `n_provider_errors` (PROTOCOL.md "Providers").

## Develop

```bash
uv sync --python 3.12
uv run python -m pytest -q                       # every external faked, about 8 s
uv run python devtools/publish.py                # build the wheel, attach it to GitHub release swarm-v<version> (needs gh)
python devtools/fake_coordinator.py --port 8765 --harnesses pi --tasks nginx-request-logging --trials 1
aidev-swarm run --coordinator http://127.0.0.1:8765
```

Dry runs with many Jobs from one account: the real coordinator allows two live leases per user, so host the
stand-in in an HF Job instead (`--host 0.0.0.0 --max-leases 25 --behind-jobs-proxy`, `expose=[8765]`; the
Jobs proxy checks the bearer token and strips it) and point `launch --coordinator https://<job>--8765.hf.jobs`
at it. `devtools/dryrun_report.py` then summarizes start latency, wall time, failures and cost.

`src/aidev_swarm/data/{grid,prices}.json` are copies of `../config/`; a test fails when they drift.
