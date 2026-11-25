# Run the evolve loop on your own agent

The workshop loop needs four things from you: tasks with a pass/fail check, three task splits, your harness in one
file, and a coding agent that proposes patches. On the toy example in this folder (an agent with a 3-step limit and a
1-second command limit, 12 small shell tasks) one run took 3 minutes, cost under one cent of Hugging Face credits and
one Claude Code proposal, and raised the held-out pass rate from 38% to 100% (4 tasks, 2 trials each, 95% interval of
the gain +25 to +100 points; GLM-5.3-Flash, run on 21 Nov 2026 by following this page).

The model stays frozen in every step below. Only the code around it changes.

## What you bring

| You bring | In the workshop | In the example (`examples/own_eval/`) |
|---|---|---|
| **1. Tasks with a pass/fail check.** Each task has a text for the agent and a check that says pass or fail. | Terminal-Bench 2.1 tasks with hidden tests; `src/aidev_evolve/evaljob.py` runs each trial and turns it into a pass/fail record | `tasks.json`: `task`, `setup`, `check` (shell commands; the check exits 0 on a pass) |
| **2. Three splits.** Search: the coding agent reads these trajectories. Validation: decides keep or drop, never shown to the coding agent. Held-out: scored once at the end. | 5 search, 8 validation, 21 held-out tasks (`src/aidev_evolve/data/splits.json`; `run --splits` takes another file) | `splits.json`: 4, 4 and 4 tasks |
| **3. The harness in one file.** Everything the coding agent may edit: prompts, the loop, tools, what the model sees of a command's output, when to stop. | `src/aidev_evolve/seed/agent.py` (62 lines; `run --seed` takes another file with the same `run(task, env)` contract) | `agent.py` (28 lines) |
| **4. A coding agent as proposer,** and a short prompt that tells it how to approach a patch and what it may edit. | Claude Code or Codex on your own plan; `src/aidev_evolve/strategies/<name>/proposer_prompt.md` | the same presets, `--strategy code-audit` by default |

The loop that connects them is the workshop's, unchanged: `src/aidev_evolve/loop.py`.

## Step 1: run the example (3 minutes)

You need what the workshop needed: uv, `uvx hf auth login`, and Claude Code or Codex logged in.

```bash
git clone https://github.com/JoelNiklaus/aidev-harness-workshop
cd aidev-harness-workshop/hands-on/part2-evolve
uv sync --python 3.12
uv run python examples/own_eval/evolve_my_agent.py --proposer claude     # or --proposer codex
```

The terminal shows what you saw in the workshop: the seed's scores, the proposed patch with its hypothesis and diff,
validation before and after, KEPT or DROPPED, and the held-out check. In my run the coding agent lifted the step, output
and timeout limits in one patch; validation went from 46% to 100% (the patch needed +38 points to count), and the
held-out check went from 38% to 100%. The run folder is `runs/own-eval-<time>/`, and
`uv run aidev-evolve show runs/own-eval-<time> --diff` prints what changed in `agent.py`.

Each trial of the example runs the model's shell commands on your machine, in a fresh temporary folder. That is fine
for these toy tasks. For your own tasks, run each trial in a container or a remote sandbox.

## Step 2: replace the four parts with yours

Copy `examples/own_eval/` and change one part at a time. Run the script after each change.

1. **`tasks.json`: your tasks and your scorer.** `task` is the text your agent gets, `setup` prepares an empty folder,
   `check` exits 0 on a pass. If your scorer is a Python function or a test suite, call it from `trial()` in
   `evolve_my_agent.py` instead of the shell check. A check that gives different answers on the same output makes
   every comparison noisier, so fix that first.
2. **`splits.json`: which task is in which split.** No task appears in two splits. Put tasks the starting agent fails
   into search and validation, otherwise there is nothing to find and nothing to measure. Keep a few tasks it passes
   in validation as guards, so a patch that breaks working behaviour gets dropped. `budget_s` is the time one task
   run may take.
3. **`agent.py`: your harness.** It defines `run(task, env)`. `env.llm(messages, tools=...)` calls the frozen model,
   `env.shell(command, timeout=...)` runs a command where the task lives, `env.time_left()` gives the seconds left.
   The loop refuses a patched `agent.py` that imports anything outside a short list of standard-library modules, that
   opens files itself, or that names a task. Those rules keep a patch general and keep it from reading the checks.
4. **`trial()` and the model in `evolve_my_agent.py`: how one task runs.** `trial()` sets up the task, calls
   `runtime.run_agent(...)` and runs the check. Swap `shell_in()` for your container or sandbox, and the `RouterLLM`
   line for your model: any function `llm(messages, tools, *, timeout, **params)` that returns the assistant message
   and the token counts works.
5. **`HEADER` in `evolve_my_agent.py`: two sentences that tell the coding agent what your agent does and what a pass
   is.** How it should approach a patch is the strategy: `uv run aidev-evolve strategy-init code-audit my-strategy`
   copies a preset, you edit `my-strategy/proposer_prompt.md`, and `--strategy my-strategy` runs it.

To score one agent file once on the workshop's Terminal-Bench tasks, without a loop (one HF Job, about a minute, well
under one cent):

```bash
uv run aidev-evolve eval --agent examples/own_eval/agent.py --tasks fix-git --trials 1
```

To run the full workshop loop on your own agent file and your own choice of Terminal-Bench 2.1 tasks, pass both to
`run`: `uv run aidev-evolve run --seed my_agent.py --splits my_splits.json`. The splits file has the keys `search`,
`validation`, `heldout` (lists of task names) and `budget_s`.

## Step 3: decide whether to believe the result

- **Read the held-out interval, not the validation curve.** Validation picked the patch, so its score is biased
  upwards. The held-out tasks were opened once. With 4 held-out tasks my interval was +25 to +100 points: the sign is
  clear, the size is not. With the workshop's 21 held-out tasks one run still carries about ±18 points.
- **Size the task set for the gap you care about.** From Part 1: about 50 tasks show a 19-point gap between two
  harnesses, about 187 tasks a 10-point gap (three trials per task, at the rerun flip rate of 0.24 measured on
  Terminal-Bench 2.1 across the four models of the Harness Leaderboard). Below 20 held-out tasks, treat a gain as a
  lead to re-measure.
- **Never let the coding agent see validation or held-out trajectories, or any check.** The loop hands it the search
  trajectories, the current `agent.py` and the history of earlier attempts, nothing else. If you add files to its work
  folder, check them against this rule.
- **Ask each patch which failures it fixes.** `PATCH.md` in `runs/<time>/candidates/c001/` names the hypothesis and
  the trajectories it came from. A patch that cannot name them fits the coding agent's theory, not your failures.
- **Look for what no trajectory shows.** In the workshop the weakness worth the most (a cleanup that killed servers
  after every command) left no failed step in any trajectory. Only the strategy that reads the harness code first
  found it. Give the coding agent both the code and the trajectories.

## When your harness is a repository, not one file

The loop here patches a single Python file. A harness spread over a repository (an existing coding agent, a product's
agent service) needs the research setup this workshop is based on:

- [Meta-Harness onboarding prompt](https://github.com/stanford-iris-lab/meta-harness/blob/main/ONBOARDING.md): paste it
  into your coding agent. It interviews you about the task, the harness interface, the evaluation, the splits and the
  budget, and writes a `domain_spec.md` you can build from. It is from the
  [Meta-Harness paper](https://arxiv.org/abs/2603.28052) (Stanford IRIS Lab).
- [JoelNiklaus/harness-optimization](https://github.com/JoelNiklaus/harness-optimization): my fork with the legal agent
  experiment from the talk (63.4% to 80.1% on Harvey's Legal Agent Benchmark with the model frozen), where the harness
  is a git repository and every candidate is a commit.

## What it costs

- The example: under one cent of HF credits per run (64 trials with 171 model calls in my run, $0.008), plus
  one or two proposals on your coding-agent plan.
- The workshop loop on Terminal-Bench tasks: about $0.3 of HF credits per run (40 runs at once on 23 Nov 2026: mean
  $0.28, at most $0.33). A run takes about 6 minutes on its own and took 12 to 15 minutes with the 40 at once.
- Your own eval: trials per iteration = (search tasks + 2 x validation tasks) for the patch + 2 x validation tasks for
  the current best, twice when a winner is re-measured. Multiply by your cost per trial before you start.
