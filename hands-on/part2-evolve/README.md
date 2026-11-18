# aidev-evolve: evolve a coding-agent harness on Terminal-Bench 2.1

Part 2 hands-on of the AI Dev NYC workshop ("Don't train the model, evolve the harness"). You start with a
59-line terminal agent (`src/aidev_evolve/seed/agent.py`) with realistic weaknesses (a 10-step limit, every
command killed after 10 s, head-only output, hidden stderr), and you pick or edit an **optimization strategy**. Your own coding agent (Claude Code or Codex)
reads its failed transcripts and patches it; an HF Job in your account scores each patch with GLM-5.3-Flash
on Terminal-Bench 2.1 tasks in HF Sandboxes; the loop keeps a patch only if it wins on validation tasks the
proposer never sees. Same model and benchmark as Part 1.

## Run it (attendee): three commands

```bash
# at home: uv installed, HF account with credits (coupon), `uvx hf auth login`, and Claude Code or Codex logged in
W="https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.4.0/aidev_evolve-0.4.0-py3-none-any.whl"
uvx --from "$W" aidev-evolve check                       # HF login, your coding agent, the workshop board (free, seconds)
uvx --from "$W" aidev-evolve strategies                  # the four presets: knobs, time, credits
uvx --from "$W" aidev-evolve run --strategy baseline --proposer claude   # or failures-first, small-diffs, two-candidates; --proposer codex|manual
uvx --from "$W" aidev-evolve replay                      # no Wi-Fi: a recorded real run, offline, no credits
```

`<WHEEL_URL>` is today `https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.4.0/aidev_evolve-0.4.0-py3-none-any.whl`
(a release asset of this public repo: no login needed; each evaluation Job downloads the same file).

`run` prints the workshop board URL (`https://agent-harness-workshop-aidev-swarm.hf.space/evolve`) and then, per
iteration: the proposer's title and one-line hypothesis, the coloured diff of `agent.py`, each trial as it finishes,
validation before -> after, KEPT or DROPPED with the fixed and broken tasks, and the HF credits spent. Every update
also goes to the board; when the Wi-Fi drops, updates wait in `runs/<time>/outbox.jsonl` and go out with the next
one (`--no-board` keeps your run private). `check --job` adds one real seed trial in an HF Job (about a minute, under
a cent). Ctrl-C, then `run --resume --run-dir runs/<time>` continues without repeating paid work.

## Your optimization strategy (0.4.0)

`--strategy` takes a preset or a folder: `aidev-evolve strategy-init small-diffs my-strategy` copies a preset; edit
`my-strategy/strategy.toml` (which transcripts the proposer reads, how much history, 1 or 2 candidates per iteration,
validation trials, keep margin z, re-measure, iterations) and `my-strategy/proposer_prompt.md` (how your coding agent
should approach a patch), then `aidev-evolve run --strategy my-strategy`. Fixed for everyone, so strategies compete
fairly: the model, 180 s per task, the splits and the 23 held-out tasks, the seed, 2 held-out trials, a 22-minute
session (a new iteration starts only if it and the held-out check still fit) and at most $2 of HF credits.

Every run ends with the seed and your final agent on the 23 held-out tasks (2 trials each, about 4-5 min, $0.30) and a
paired gain with a 95% CI. One run's interval is about ±17 points, so the board ranks **strategies**, pooling every
run of the same strategy (and every edit of a preset), with Welch intervals and leaderboard-style ties
(`aidev-evolve rank runs/*` does the same offline). Details: DESIGN.md, "Phase 3".

Measured (PROGRESS.md): about 3 min for the seed, 4 min per iteration (7 with a re-measure), 2 iterations per session,
$0.8-1.3 per run.

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
