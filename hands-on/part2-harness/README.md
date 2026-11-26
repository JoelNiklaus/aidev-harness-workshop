# Part 2: fix a harness and watch its score go up

Setup and the challenge are in the [main README](../../README.md). Run `aidev-evolve eval` and your coding agent in this folder.

| File | What it is |
|---|---|
| `agent.py` | the harness, the only file that changes |
| `CONTRACT.md` | what `agent.py` may use (`run(task, env)`, `env.llm`, `env.shell`) and what the workshop fixes |
| `AGENTS.md`, `CLAUDE.md` | what your coding agent reads when it starts in this folder: it acts as a teacher, shows you where to look and makes the changes you decide |
| `chart.html` | score by version; open it with a double-click |
| `traces/vN/`, `versions/vN/agent.py`, `versions.json` | written by `eval`: the trajectories, the evaluated file and the score of every version |
| `docs/` | how `eval` works, and how to build the same loop around your own agent |
| `hints/` | only if you are stuck: sample questions for your coding agent, and three hints that reveal one flaw each |
