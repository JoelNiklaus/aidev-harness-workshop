# Take home

## 1. Check your result on tasks nobody read

```bash
aidev-evolve eval --heldout
```

This scores your latest version and version 1 on 21 other tasks, in one Job (about 3 minutes, about 15 cents).
You fixed the harness while reading the trajectories of the 8 validation tasks, so those 8 scores flatter the harness a little.
The held-out score is the number to report. Measured on 25 Nov 2026 in two sessions: version 1 passed 6 of 21 (29%),
version 4 passed 15 of 21 (71%).

## 2. Go on with this harness

Three flaws are planted, but the harness has more to gain. In the workshop your coding agent was a teacher that
gave hints only. To let it work on its own now, delete the section "Your role: a helpful teacher" and the two
sections after it from `AGENTS.md`. Ideas: what the model sees of a long command output, what it is told when a command fails, how the
conversation is kept when it gets long. Every `eval` adds a bar, and `versions/vN/agent.py` keeps every version, so
you can always go back:

```bash
cp versions/v4/agent.py agent.py
```

## 3. Use the loop on your own agent

The loop of this session has four parts. You can build each of them for any agent.

1. **Readable trajectories.** After every run, write one file per task that a person or a coding agent can read: what the
   model saw, what it did, what came back, and how the run ended. Add a one-table index. Most harness flaws show up
   in the difference between what a tool returned and what the model received.
2. **A fixed evaluation set and a held-out set.** Pick a small set of tasks with automatic checks for the loop
   (8 here) and keep a second set that you never read (21 here).
3. **One command.** It evaluates the current version, stores the version and its trajectories, and adds a point to a chart.
   When it takes more than a few minutes, you stop running it.
4. **One change per version.** Ask the coding agent for the evidence in the trajectories first, then change one thing.
   With one change per version, the chart tells you what each change was worth.

Pitfalls:

- A small set is noisy. With 8 tasks and one try each, a score moves by one or two tasks between runs. Use more
  tries per task before you trust a small gain.
- A fix for one task is not a harness fix. Look for a pattern across several failed tasks.
- Do not read the held-out trajectories, and do not run the held-out check after every change.

The evaluation tool is in [`../part2-evolve`](../../part2-evolve) of this repository. To evaluate another harness on
the same tasks, copy this folder, replace `agent.py` with any file that follows `CONTRACT.md`, delete `versions.json`,
`versions/` and `traces/`, and run `aidev-evolve eval` there.

In practice this loop runs automated: a proposer model writes the changes, and a search keeps the versions that score
higher. Two places to start: [meta-harness](https://github.com/stanford-iris-lab/meta-harness) (you can add your own
benchmark) and [harness-optimization](https://github.com/JoelNiklaus/harness-optimization).
