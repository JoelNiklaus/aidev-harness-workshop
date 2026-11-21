# aidev-evolve: evolve a coding-agent harness on Terminal-Bench 2.1

Part 2 hands-on of the AI Dev NYC workshop ("Don't train the model, evolve the harness"). You start with a
62-line terminal agent (`src/aidev_evolve/seed/agent.py`) with realistic weaknesses (a 10-step limit, the task
text cut at 1,000 characters, every program a command leaves running killed after it, every command killed after
10 s, head-only output, hidden stderr), and you pick or edit an **optimization strategy**. Your own coding agent (Claude Code or Codex)
reads its failed trajectories and patches it; an HF Job in your account scores each patch with GLM-5.3-Flash
on Terminal-Bench 2.1 tasks in HF Sandboxes; the loop keeps a patch only if it wins on validation tasks the
proposer never sees. Same model and benchmark as Part 1.

## Run it (attendee): three commands

```bash
# at home: uv installed, HF account with credits (coupon), `uvx hf auth login`, and Claude Code or Codex logged in
W="https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.5.3/aidev_evolve-0.5.3-py3-none-any.whl"
uvx --from "$W" aidev-evolve check                       # HF login, your coding agent, the workshop board (free, seconds)
uvx --from "$W" aidev-evolve strategies                  # the six presets: knobs, time, credits
uvx --from "$W" aidev-evolve run --block left            # your seating block: left, middle or right (or --strategy <preset>); --proposer codex|manual
uvx --from "$W" aidev-evolve show                        # afterwards: your final agent.py (--diff: what changed against the seed)
uvx --from "$W" aidev-evolve replay                      # no Wi-Fi: a recorded real run, offline, no credits
```

In the room each seating block runs one strategy, so the board has enough runs of each to rank them:
`--block left` = `baseline`, `--block middle` = `every-weakness`, `--block right` = `code-audit` (A, B, C work too).
Without `--block` or `--strategy`, `run` uses `code-audit`. The Part 2 start page of the board (`/evolve/join`)
shows the three commands with Copy buttons.

`<WHEEL_URL>` is today `https://github.com/JoelNiklaus/aidev-harness-workshop/releases/download/evolve-v0.5.3/aidev_evolve-0.5.3-py3-none-any.whl`
(a release asset of this public repo: no login needed; each evaluation Job downloads the same file).

`run` prints the workshop board URL (`https://agent-harness-workshop-aidev-swarm.hf.space/evolve`) and then, per
iteration: the proposer's title and one-line hypothesis, the coloured diff of `agent.py`, each trial as it finishes,
validation before -> after, KEPT or DROPPED with the fixed and broken tasks, and the HF credits spent. Every update
also goes to the board; when the Wi-Fi drops, updates wait in `runs/<time>/outbox.jsonl` and go out with the next
one (`--no-board` keeps your run private; `--board <url>` or `export AIDEV_BOARD=<url>`, the setting `aidev-swarm` reads
too, posts to another board, e.g. a staging Space for a dry run). `check --job` adds one real seed trial in an HF Job (about a minute, under
a cent). Ctrl-C, then `run --resume --run-dir runs/<time>` continues without repeating paid work.

A run ends with the path of `runs/<time>/final_agent.py`, the best agent with every kept patch. `aidev-evolve show`
prints the newest run's final agent, `show --diff` only what changed against the seed, and both end with a text check
of the seed's three measured weaknesses (step limit, task text cut, stray-process cleanup: still there or removed).

To run the loop on your own agent and tasks, read [APPLY-TO-YOUR-AGENT.md](APPLY-TO-YOUR-AGENT.md).

## Your optimization strategy (0.4.0; six presets since 0.5.0)

`--strategy` takes a preset or a folder (default `code-audit`, the measured winner below):
`aidev-evolve strategy-init small-diffs my-strategy` copies a preset; edit
`my-strategy/strategy.toml` (which trajectories the proposer reads, how much history, 1 or 2 candidates per iteration,
validation trials, keep margin z, re-measure, iterations) and `my-strategy/proposer_prompt.md` (how your coding agent
should approach a patch), then `aidev-evolve run --strategy my-strategy`. Fixed for everyone, so strategies compete
fairly: the model, 180 s per task, the splits and the 21 held-out tasks, the seed, 2 held-out trials, a 22-minute
session (a new iteration starts only if it and the held-out check still fit) and at most $2 of HF credits.

Every run ends with the seed and your final agent on the 21 held-out tasks (2 trials each, about 4-5 min, $0.25) and a
paired gain with a 95% CI. One run's interval is about ±18 points (20 runs: half-widths 14 to 24), so the board ranks **strategies**, pooling every
run of the same strategy (and every edit of a preset), with Welch intervals and leaderboard-style ties
(`aidev-evolve rank runs/*` does the same offline). Details: DESIGN.md, "Phase 3" and "Phase 4".

Measured on 20 real runs (PROGRESS.md, Phase 4): `code-audit` (read agent.py line by line, then fix every harmful
line in one patch) reached 70% on the held-out tasks, ahead of `every-weakness` (58%) and `baseline` (50%; seed 35%),
and the board would rank it first on its own. About 3 min for the seed, 4-5 min per iteration (7-8 with a re-measure),
2 iterations per session, $0.4-0.8 per run.

Each attendee's GLM-5.3-Flash calls go to one provider picked from the workshop's shared list by a hash of the HF
username (75% novita, 25% zai-org, as in Part 1); a busy provider (429/5xx) falls back to the next one, and every
board event records the provider.

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
uv run python -m pytest -q          # all externals faked, about 10 s
uv run python devtools/publish.py   # build the wheel, attach it to GitHub release evolve-v<version> (needs gh)
uv run aidev-evolve eval --agent src/aidev_evolve/seed/agent.py --tasks fix-git,regex-log --trials 1
python devtools/fake_board.py       # the /evolve board stand-in on :8766
```
