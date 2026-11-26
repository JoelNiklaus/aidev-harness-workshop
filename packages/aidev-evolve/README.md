# aidev-evolve: the evaluation tool of Part 2

Part 2 hands-on of the AI Dev NYC workshop ("Don't train the model, evolve the harness"). Attendees work in
[`part2-harness`](../../part2-harness) (setup and the challenge: the main README): a 172-line terminal agent (`agent.py`, the *harness*: the
loop around a fixed model) with three planted flaws. They decide how to investigate and what to change, and their own coding agent
reads the trajectories and edits; `aidev-evolve eval` scores the new version with GLM-5.3-Flash on 8 Terminal-Bench 2.1 tasks in an HF Job in their
account; a local chart shows the score by version. This folder is the tool behind that command.

## What `eval` does

```bash
# once, from the root of the clone (the main README's setup): aidev-lb, aidev-evolve and hf as uv tools
uv tool install ./packages/aidev-lb --with-executables-from "aidev-evolve @ ./packages/aidev-evolve" --with-executables-from huggingface-hub
cd part2-harness
aidev-evolve eval             # scores agent.py as the next version (the first run records version 1)
aidev-evolve eval --heldout   # the latest version and version 1 on 20 tasks nobody read, in one Job
```

The Job downloads the wheel of this version from the Hugging Face repository `launch.WHEEL_REPO`, with one more try
when the download fails; `--package-url` names another wheel. Nothing on the path of a trial comes from GitHub: the
task folders travel in the wheel (`data/tb21-tasks.tar.gz`, built by `devtools/build_assets.py`), and `nogithub.py`
(the same file as in the Part 1 runner) sends the verifier's downloads to Astral's mirror and repeats a sandbox call
the API refused with 503. The trials of one version are stopped at a wall limit (`evaljob.wall_limit_s`).
`<WHEEL_URL>` is today `https://huggingface.co/datasets/agent-harness-workshop/aidev-workshop-wheels/resolve/main/evolve/aidev_evolve-0.7.0-py3-none-any.whl`
(a public repository).

`eval` prints the Job page at once, one line per trial as results arrive, then `vN: K/8 passed`. A folder with
`agent.py` and no `versions.json` gets version 1 from its first `eval`. An
unchanged `agent.py` is not evaluated again. A trial lost to infrastructure (sandbox, provider) is not counted as a fail.
The agent time per task is 120 s (`data/splits.json`, `eval_budget_s`).

Measured on 26 Nov 2026 (provider deepinfra, 8 tasks x 4 trials per row, 120 s per task, one Job, 0 infrastructure errors):

| Version | Passed | Slowest trial of an evaluation |
|---|---|---|
| v1, as shipped | 5/32 (16%) | 47 to 97 s |
| task text no longer cut | 10/32 (31%) | 48 to 106 s |
| turn limit fixed only | 13/32 (41%) | 68 to 150 s |
| both | 20/32 (63%) | 73 to 156 s |
| all three flaws fixed | 30/32 (94%) | 70 to 85 s |

Held-out (20 tasks x 2 trials): v1 13/40 (33%), all three fixed 27/40 (68%). The validation task `build-pmars` came from the
held-out tasks on 26 Nov 2026, in place of a slow one (data/splits.json has the reason; devtools/ladder.py `--tasks`
measured the candidates).

One evaluation is 8 trials, so a version's score moves by one or two tasks between runs.

GLM-5.3-Flash calls go to one provider picked from the packaged list (`data/providers.json`: together, coreweave,
deepinfra, wafer at weight 2, particle-ai at weight 1; load-tested on 25 Nov 2026) by a hash of the HF username; a busy
provider (429/5xx) falls back to the next one, and a provider that refuses the model (HTTP 400 "not supported by
provider") is left out for the rest of the Job.

## What your coding agent may change

Anything in `agent.py` that a harness controls: prompts, the loop, tools, how command output is shown to the
model, when to stop, how to recover. Fixed: the model and its settings, the time per task, the contract
(`run(task, env)`, `env.llm`, `env.shell`; see `part2-harness/CONTRACT.md`). `eval` refuses an `agent.py` that does
not compile or has no `run(task, env)` before any credits are spent.

## Develop

In the workshop's source repository: the public repository ships only `src`, `pyproject.toml` and this file.

```bash
uv sync --python 3.12
uv run python -m pytest -q -n 4     # all externals faked, about 20 s
uv run python devtools/publish.py   # build the wheel and upload it to the Hugging Face repository launch.WHEEL_REPO
# measure several agent files in one Job (submit returns at once; collect reads the Job log):
uv run python devtools/ladder.py submit --name val --split validation --trials 4 v1=harness v2=harness:task v3=harness:task+steps v4=harness:task+steps+cleanup
uv run python devtools/ladder.py collect --name val
```

An agent for the ladder is a file, or `harness` (`part2-harness/agent.py`) with any of its three flaws fixed
(`steps`, `task`, `cleanup`); `--budget-s` sets another agent time. `tests/test_harness.py` shows what each
fix changes. An unreleased wheel can be measured from a private Hub file:
`--package-url https://huggingface.co/datasets/<you>/<repo>/resolve/main/<wheel>` (the Job downloads it with its own token).
