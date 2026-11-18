# aidev-evolve: evolve a coding-agent harness on Terminal-Bench 2.1

Part 2 hands-on of the AI Dev NYC workshop ("Don't train the model, evolve the harness"). You start with a
58-line terminal agent (`src/aidev_evolve/seed/agent.py`) with four realistic weaknesses. Your own coding agent (Claude Code or Codex)
reads its failed transcripts and patches it; an HF Job in your account scores each patch with GLM-5.3-Flash
on Terminal-Bench 2.1 tasks in HF Sandboxes; the loop keeps a patch only if it wins on validation tasks the
proposer never sees. Same model and benchmark as Part 1.

## Run it (attendee)

```bash
# at home: uv installed, HF account with credits, `uvx hf auth login`, and one of
#   Claude Code logged in (`claude -p "say ok"` answers) or Codex logged in (`codex exec "say ok"`)
uvx --from "<WHEEL_URL>" aidev-evolve run --proposer claude --model sonnet     # or --proposer codex
uvx --from "<WHEEL_URL>" aidev-evolve run --proposer manual                    # any coding agent, by paste
uvx --from "<WHEEL_URL>" aidev-evolve replay                                   # no Wi-Fi: a recorded real run
```

`<WHEEL_URL>` is today `https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.2.1/aidev_evolve-0.2.1-py3-none-any.whl`
(a release asset of this public repo: no login needed; each evaluation Job downloads the same file).

Defaults (Phase 2, config B): 6 iterations, 180 s per task, validation 2 trials per task, keep at z = 1.5 plus one
re-measure, and a held-out check with 2 trials per task. Measured: about 4 min and $0.14 per iteration, a full run
about $1.0-1.3 (PROGRESS.md). The terminal
shows the running HF spend. Results go to `runs/<time>/` (see DESIGN.md). `--board <room URL>` posts your
progress to the room's `/evolve` page (PROTOCOL.md; the room board is the Part 1 coordinator,
`https://agent-harness-workshop-aidev-swarm.hf.space`, once it serves `/evolve`).

## What your coding agent may change

Anything in `agent.py` that a harness controls: prompts, the loop, tools, how command output is shown to the
model, when to stop, how to recover. Fixed: the model and its settings, the 180 s per task, the contract
(`run(task, env)`, `env.llm`, `env.shell`; see `proposer.CONTRACT`). An audit refuses network or file
access from agent.py and any benchmark task name; a preflight run refuses a crashing agent before any
credits are spent.

## Develop

```bash
uv sync --python 3.12
uv run python -m pytest -q          # all externals faked, about 5 s
uv run python devtools/publish.py   # build the wheel, attach it to GitHub release evolve-v<version> (needs gh)
uv run aidev-evolve eval --agent src/aidev_evolve/seed/agent.py --tasks fix-git,regex-log --trials 1
python devtools/fake_board.py       # the /evolve board stand-in on :8766
```
