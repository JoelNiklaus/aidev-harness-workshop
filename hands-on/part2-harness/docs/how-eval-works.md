# How the evaluation works

## What runs where

On your laptop, `aidev-evolve eval`:

1. reads `agent.py` and checks that it loads and defines `run(task, env)`;
2. starts one Hugging Face Job in your account (machine type `cpu-basic`). The text of `agent.py` travels in a Job
   variable and your token in a Job secret;
3. reads the Job's log while it runs and prints one line per finished task;
4. writes `traces/vN/`, `versions/vN/agent.py`, `versions.json` and `chart.html`.

In the Job:

1. the evaluation tool is installed from the release wheel of this repository;
2. every task gets its own Hugging Face Sandbox, a container with the task's files (Terminal-Bench 2.1);
3. your `run(task, env)` is called once per task, all 8 tasks at the same time. `env.shell` runs a command in the
   task's sandbox. `env.llm` calls GLM-5.3-Flash through Hugging Face Inference Providers, with low reasoning effort;
4. when `run` returns, or after 120 seconds, the task's hidden tests check the sandbox. A task passes or fails.

The model provider is picked from a fixed list (together, coreweave, deepinfra, wafer, particle-ai) by a hash of your
username, so the room is spread over five providers. When your provider is busy, the call goes to the next one.

A task that was lost to infrastructure (a sandbox or a provider error) is not counted as a fail. The score then reads
`5/7` in place of `5/8`.

## What it costs

Everything is billed to your Hugging Face account. Measured on 26 Nov 2026:

| Run | Time | Cost |
|---|---|---|
| `eval`, version 1 | 60 to 110 s | $0.01 |
| `eval`, later versions | 60 to 170 s | $0.01 to $0.04 |
| `eval --heldout` (20 tasks, two versions) | about 180 s | $0.13 |

A whole session (four evaluations and the held-out check) costs about $0.20.
Most of the cost is the model calls. The Job machine costs $0.01 per hour and each sandbox $0.02 per hour.

The time of one evaluation is the time of its slowest task. A task that uses its whole 120 seconds makes the
evaluation take about 2.5 minutes.

## What the trajectory files contain

`traces/vN/INDEX.md` is one table for version N: per task the result, how the run ended (`returned`, `timeout` or
`error`), the number of model calls, the agent time next to the time budget, and the file name of the trajectory.
Failed tasks come first.

`traces/vN/<task>__1.md` is the trajectory of one task:

- the head line: result, how the run ended, agent time, model calls, tokens;
- the task text;
- every step in order: what the model said, the tool call it made, the command as the sandbox ran it, the exit code,
  the duration, how many characters the command printed, the output (long outputs keep the start and the end), and how
  many characters of it the model received;
- a note when the model did not receive what the harness was given;
- the end: how the run ended and the result of the hidden tests.

The hidden tests themselves are not in the trajectories. The held-out tasks get no trajectories at all, so nobody can tune a
harness on them.

`versions.json` holds per version the score, the result per task, the time, the cost and the link of the Job.
