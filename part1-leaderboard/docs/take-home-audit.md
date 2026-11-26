# Take-home: did my agent look up the answer?

A pass only counts if the agent solved the task itself. Terminal-Bench tasks are public, so an agent with
network access can find a solution online. An audit checks each passed trial for that. It is not part of the
workshop, and nothing here is built. This page describes how to build it.

## What to check

For each passed trial, read the agent's trajectory (every command it ran and every page it fetched) and flag:

1. A fetch of the benchmark's own repository or of a page that contains the task's solution.
2. A search for the task name or for a string from the task's tests.
3. A read of the test files or of a solution file inside the sandbox, when the task does not ask for it.
4. A very short trial that writes the final answer without any exploration.

## How to build it

1. Keep the trajectories. Harbor writes one folder per trial inside the cloud job, with the agent's log. Copy
   these folders to your bucket at the end of the job (the runner is in `packages/aidev-lb`,
   `src/aidev_lb/job.py`).
2. First pass with plain rules: search each trajectory for the task name, `terminal-bench`, `github.com`, and
   the names of the test files.
3. Second pass with a model: give it the task text and the trajectory, and ask for one of `clean`,
   `suspicious`, `looked up`, with the line that shows it. Read every `suspicious` and `looked up` verdict
   yourself.
4. Add a column to your leaderboard: pass rate with flagged trials counted as fails.

## What to expect

Most flags are harmless, for example a package index. Judge the flagged line, not the count. Report both pass
rates, with and without flagged trials.
