# Agent harnesses workshop (AI Dev NYC 2026)

**After Part 1:** tonight, run one trial on your own account (section "After Part 1" below). For Part 2, bring Claude Code or Codex, logged in, and run `aidev-evolve check` tonight (section "Part 2 in the room").

Code for the two hands-on sessions of the Agent Harnesses workshop at AI Dev NYC (Mon 30 Nov and Tue 1 Dec 2026). In Part 1 we evaluate 10 coding-agent harnesses on 20 Terminal-Bench 2.1 tasks with the same model (GLM-5.3-Flash): every attendee's Hugging Face Job claims trials from a shared grid, and a live board shows pass rate and cost per harness as results arrive.

The point of Part 1: a benchmark score is a property of the model and its harness together. With the model fixed, swapping the harness moves the pass rate by double-digit points and the cost per solved task by several times, so a leaderboard that lists only the model hides half of what it measured.

## Before the workshop (10 minutes)

You need a Hugging Face account with credits and a fine-grained token with the permissions "Make calls to Inference Providers" and "Start and manage Jobs". Then install [uv](https://docs.astral.sh/uv/) and run:

```bash
uvx hf auth login
uvx --from "https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/swarm-v0.4.3/aidev_swarm-0.4.3-py3-none-any.whl" aidev-swarm check
```

`check` makes one GLM-5.3-Flash call, runs one tiny HF Job and one command in an HF Sandbox (15 to 35 seconds in our runs, well under one cent) and ends with `ready: ...`. It claims no workshop task.

## Part 1 in the room

Start page with every step, the commands to copy and the usual fixes: https://niklaus.ai/aidev-harness-workshop/1

```bash
uvx --from "https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/swarm-v0.4.3/aidev_swarm-0.4.3-py3-none-any.whl" aidev-swarm launch
```

`launch` submits one cpu-basic HF Job in your own namespace and returns in about a second. The Job runs up to 8 trials at once (`--parallel N` to change it; a full room gets fewer): each slot claims a cell (harness, task, trial), runs it with [Harbor](https://github.com/laude-institute/harbor) in its own HF Sandbox, submits the result to the board and claims the next one. Your laptop can close. Or paste [`AGENT_PROMPT.md`](hands-on/part1-swarm/runner/AGENT_PROMPT.md) into your coding agent and let it run and watch the Job.

## After Part 1: one trial on your own account

Run one trial without the workshop's grid, for example to compare harnesses on this model. It installs Harbor and the Harness Leaderboard's harness setup on your laptop (about 290 MB) and runs the trial in an HF Sandbox in your namespace:

```bash
uvx --from "aidev-swarm[run] @ https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/swarm-v0.4.3/aidev_swarm-0.4.3-py3-none-any.whl" \
  aidev-swarm run --local --harness opencode --task nginx-request-logging
```

Change `--harness` (for example `pi`, `openclaw`, `codex`) or `--task` (any task in [`config/grid.json`](hands-on/part1-swarm/config/grid.json)).

## Part 2 in the room

Start page with every step, the commands to copy and the usual fixes: https://niklaus.ai/aidev-harness-workshop/2

```bash
W="https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.6.0/aidev_evolve-0.6.0-py3-none-any.whl"
uvx --from "$W" aidev-evolve check                                     # HF login, your coding agent, the workshop board
uvx --from "$W" aidev-evolve run --block left                          # your seating block: left, middle or right; add --proposer codex for Codex
uvx --from "$W" aidev-evolve show                                      # afterwards: your final agent.py (--diff: what changed)
uvx --from "$W" aidev-evolve replay                                    # a recorded real run, offline
```

Each seating block runs one optimization strategy (`left` = `baseline`, `middle` = `every-weakness`, `right` = `code-audit`), so the board has enough runs of each to rank them by their held-out gain. `run` prints the workshop board's address and streams every iteration to it, and it ends with the path of your final `agent.py`. The model, time budget, tasks and held-out set are the same for everyone. `aidev-evolve strategies` lists all six strategies; `aidev-evolve strategy-init <preset> my-strategy` copies one to edit (its knobs and the proposer prompt) and `--strategy my-strategy` runs it. Use `--proposer manual` to paste the prompt into any coding agent. Leave the laptop open: a run takes about 20 minutes and ends with a held-out check.

## After Part 2: the loop on your own agent

[`APPLY-TO-YOUR-AGENT.md`](hands-on/part2-evolve/APPLY-TO-YOUR-AGENT.md) (as a page: https://niklaus.ai/aidev-harness-workshop/guide) shows what you need (tasks with a pass/fail check, three splits, your harness in one file, a coding agent) and a toy example that runs in 3 minutes for under one cent.

## Layout

- [`hands-on/part1-swarm/runner`](hands-on/part1-swarm/runner): the attendee CLI `aidev-swarm` (`check`, `launch`, `run`). `run --local --harness pi --task nginx-request-logging` runs one trial without the workshop's coordinator, for use after the workshop.
- [`hands-on/part1-swarm/coordinator`](hands-on/part1-swarm/coordinator): the FastAPI app behind the board (claim queue with leases, results, the live boards and the start pages `/join` and `/evolve/join`).
- [`hands-on/part1-swarm/replay`](hands-on/part1-swarm/replay): pre-run reference trials from the [Harness Leaderboard](https://huggingface.co/spaces/joelniklaus/harness-leaderboard) (every trial's trajectory is public in [its bucket](https://huggingface.co/buckets/joelniklaus/harness-leaderboard)), shown next to the workshop's results and replayed when the venue network fails.
- [`hands-on/part1-swarm/config`](hands-on/part1-swarm/config): the grid (model, harnesses, tasks, trials) and the list prices behind every cost on the board.
- [`hands-on/part1-swarm/PROTOCOL.md`](hands-on/part1-swarm/PROTOCOL.md): the HTTP protocol between runner and coordinator.
- [`hands-on/part2-evolve`](hands-on/part2-evolve): Day 2, the CLI `aidev-evolve`. Your coding agent (Claude Code or Codex) proposes patches to a small terminal agent, an HF Job in your account scores each patch with GLM-5.3-Flash on Terminal-Bench 2.1 tasks, and the loop keeps a patch only if it wins on tasks the proposer never sees. [`examples/own_eval`](hands-on/part2-evolve/examples/own_eval) runs the same loop on a toy agent with its own tasks.

## Develop

```bash
cd hands-on/part1-swarm/runner && uv run python -m pytest -q       # every external call faked, seconds
cd hands-on/part1-swarm/coordinator && uv run python -m pytest -q
cd hands-on/part1-swarm/coordinator && uv run coordinator --replay   # the board offline, http://127.0.0.1:7860
cd hands-on/part2-evolve && uv run python -m pytest -q
```

## License

Apache-2.0, see [LICENSE](LICENSE).

The 18-minute target is measured from launch to the held-out result. It has not yet been verified with 40 concurrent workshop runs. If the result is late, use `aidev-evolve replay` for the debrief.
