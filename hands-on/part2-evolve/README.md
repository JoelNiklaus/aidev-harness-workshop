# aidev-evolve: evolve a coding-agent harness on Terminal-Bench 2.1

Part 2 hands-on of the AI Dev NYC workshop ("Don't train the model, evolve the harness"). You start with a
58-line terminal agent (`src/aidev_evolve/seed/agent.py`) with four realistic weaknesses. Your own coding agent (Claude Code or Codex)
reads its failed transcripts and patches it; an HF Job in your account scores each patch with GLM-5.3-Flash
on Terminal-Bench 2.1 tasks in HF Sandboxes; the loop keeps a patch only if it wins on validation tasks the
proposer never sees. Same model and benchmark as Part 1.

## Run it (attendee): three commands

```bash
# at home: uv installed, HF account with credits (coupon), `uvx hf auth login`, and Claude Code or Codex logged in
W="https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.3.0/aidev_evolve-0.3.0-py3-none-any.whl"
uvx --from "$W" aidev-evolve check                       # HF login, your coding agent, the room board (free, seconds)
uvx --from "$W" aidev-evolve run --proposer claude       # or --proposer codex; --proposer manual: paste into any agent
uvx --from "$W" aidev-evolve replay                      # no Wi-Fi: a recorded real run, offline, no credits
```

`<WHEEL_URL>` is today `https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.3.0/aidev_evolve-0.3.0-py3-none-any.whl`
(a release asset of this public repo: no login needed; each evaluation Job downloads the same file).

`run` prints the room board URL (`https://agent-harness-workshop-aidev-swarm.hf.space/evolve`) and then, per
iteration: the proposer's title and one-line hypothesis, the coloured diff of `agent.py`, each trial as it finishes,
validation before -> after, KEPT or DROPPED with the fixed and broken tasks, and the HF credits spent. Every update
also goes to the board; when the Wi-Fi drops, updates wait in `runs/<time>/outbox.jsonl` and go out with the next
one (`--no-board` keeps your run private). `check --job` adds one real seed trial in an HF Job (about a minute, under
a cent). Ctrl-C, then `run --resume --run-dir runs/<time>` continues without repeating paid work.

Defaults (Phase 2, config B): 6 iterations, 180 s per task, validation 2 trials per task, keep at z = 1.5 plus one
re-measure, and a held-out check with 2 trials per task. Measured: about 4 min and $0.14 per iteration, a full run
about $1.0-1.3 (PROGRESS.md). Results go to `runs/<time>/` (see DESIGN.md).

## Shared room exercise: offline fallback

This local kit gives each attendee the same seed. It does not implement a live tournament, shared submission, or a shared champion.
`room-init` and `room-diff` ship in the wheel since 0.3.0.

```bash
# With this checkout's environment installed, before leaving home:
uv run aidev-evolve room-init room-proposal
# Edit room-proposal/agent.py with your own coding agent or editor.
uv run aidev-evolve room-diff room-proposal
uv run aidev-evolve replay --speed 100
```

Participants share the printed base hash, diff, hypothesis, and trade-offs with the facilitator through an agreed channel.
The facilitator reviews the patch as text in a viewer that does not interpret terminal escape sequences. The hash detects accidental base changes. It is not an identity or security proof.
The replay is historical, not a score of an attendee's patch. Cache the built wheel and dependencies on the presentation laptop before going offline.

**Safety:** Do not evaluate untrusted room patches with `run` or `eval`. The existing evaluator executes candidate Python in a Job with credentials. Its static audit is not a security boundary. A live shared evaluator requires independent credential isolation and disposable restricted execution before use.

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
