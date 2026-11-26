# aidev-lb (Part 1 runner)

`aidev-lb run --harness pi --harness qwen-code` starts **one HF Job per harness in your own namespace**. Each
Job runs its harness on 5 Terminal-Bench 2.1 tasks at once, every trial in its own HF Sandbox, with
GLM-5.3-Flash at low effort. The command prints each Job page, then one line per trial event read from the Job
log, and keeps the results in `results.json`. There is no shared server: your HF credits pay for your Jobs
(cpu-basic), the sandboxes and the model calls.

## Attendee: run it

```bash
# once: uv installed, HF account with credits, and a token of type "Write", saved with `hf auth login`
uv tool install --with-executables-from huggingface-hub ./packages/aidev-lb   # from the clone: aidev-lb and hf
aidev-lb check          # one model call, one tiny Job, one sandbox command (< $0.01)
aidev-lb harnesses      # the 10 harnesses: fast, medium or slow to install
aidev-lb run --harness pi
aidev-lb run --harness qwen-code --harness openclaw
```

The Jobs install the released wheel `<WHEEL_URL>`.
`<WHEEL_URL>` is today `https://huggingface.co/datasets/agent-harness-workshop/aidev-workshop-wheels/resolve/main/lb/aidev_lb-0.5.0-py3-none-any.whl`
(a public repository; `run --package dist/aidev_lb-0.5.0-py3-none-any.whl` sends a local wheel to the Jobs as a
Job secret instead). The laptop install is light (httpx and huggingface_hub); Harbor and the harness wiring
install only inside the Job (the `run` extra).

- `run --tasks N` takes the first N of the 10 tasks in `data/tasks.json` (default 5).
- `run` ends when every trial has a result or an infrastructure error. While it waits it prints one line every
  30 s (`still waiting: N trials (K sandboxes queued)`). After 15 minutes it cancels the Jobs still running and
  their sandboxes. A trial without a result has none: the command prints one line that says so. A repeated run replaces the older result of the same harness and task.
- `run` also keeps the results in the public bucket `<you>/harness-results` (created if missing; `--bucket NAME`
  for another name): the Jobs write one file per cell (`results/<cell_id>.json`) and the laptop writes the
  aggregate `results.json`, readable without a token at
  `https://huggingface.co/buckets/<you>/harness-results/resolve/results.json`. A token that cannot write the
  bucket gives one message, and the run goes on with the local file only. `--no-bucket` skips the bucket.
  Ctrl-C publishes the results so far.
- Every record in `results.json` has `source: "measured"`. The attendee's page bundles made-up records with
  `source: "sample"` (`part1-leaderboard/sample-results.json`) and shows them as mock rows.

## How it works

- `cli.py`: the laptop commands. `run` submits the Jobs in parallel, follows each Job log (a broken stream is
  reopened and replayed without duplicates), and writes each result as it arrives.
- `launch.py`: the Job (image `ghcr.io/astral-sh/uv:python3.12-bookworm`, cpu-basic, the token only as the Job
  secret `HF_TOKEN`). It installs this version's wheel and runs `aidev-lb job --harness H --tasks a,b,c`.
- `job.py`: inside the Job. One short model call per provider finds the first that answers; then all trials
  start in threads. A trial whose every model call ended in a provider error runs again on the next provider.
- `events.py`: the `AIDEV-LB {json}` lines the Job prints (sandbox_started, harness_installed, agent_finished,
  result, infra, retry, provider) and the plain line for each.
- `store.py`: the results store (`write_result`, `read_all`, and `publish` for the bucket's aggregate). Backend
  "log", always on: the Job prints the result on its log, the laptop keeps `results.json`. Backend "bucket",
  the default: the same records in the attendee's public bucket.
- `providers.py`: the provider list (`data/providers.json`), the attendee's order (a weighted hash of the HF
  username), and what counts as a provider error: 429, 5xx, and refusals (400 "not supported by provider", 401,
  403, 404). `result.py` turns a trial folder into a Result and names infrastructure errors: a trial that ended
  with an error before any model call, or whose every model call was a provider error, is never a fail.
- `trial.py` reuses the Harness Leaderboard's wiring (`hlb`, pinned): per-harness Harbor configs, the HF
  Sandbox environment and the in-sandbox meter. Harnesses get a dummy key; only the meter holds your token.
  The meter sends a call its provider answers with 429 on to the attendee's next provider.
- `catalog.py`: the packaged data files.

## Develop

In the workshop's source repository: the public repository ships only `src`, `pyproject.toml` and this file.

```bash
uv sync --python 3.12
.venv/bin/python -m pytest -q                    # every external faked, a few seconds
.venv/bin/python devtools/build_data.py          # rebuild ../../part1-leaderboard/sample-results.json
uv build --wheel --out-dir dist                  # the wheel `run --package` takes
uv run python devtools/publish.py                # build the wheel and upload it to the Hugging Face repository launch.WHEEL_REPO
```
