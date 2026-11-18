# Requests from the coordinator owner (Part 1 coordinator / room board)

Write-only for the coordinator helper; the Part 2 owner decides. Contract: `PROTOCOL.md`.

## 2026-11-18 — /evolve is live on the coordinator; two small client changes

1. **Send the patch's hypothesis with each iteration event.** The board's patch feed shows one line of it
   under the title, and the "most common fixes" tally falls back to it when the title names no known fix.
   In `loop.py` `_finish`, add to the `kind: "iteration"` event:
   `"hypothesis": h.get("hypothesis"),`
   (string, the server keeps the first line, drops a leading `Hypothesis:` and cuts to 120 chars for display;
   up to 1,000 chars accepted). Without it the feed shows the title only.
2. **Optional:** treat HTTP 413 like the other non-retryable 4xx (it already is). The server now *shortens*
   overlong text fields (title 200, diff 8,000, task names 100, at most 100 fixed/broke tasks) instead of
   answering 422, so a verbose PATCH.md no longer loses an iteration's point on the race chart. A body over
   64 KB is refused with 413.

Nothing else is needed: the client's outbox, retry on 401/408/429/5xx and `(user, run_id, kind, iteration)`
idempotency match the server. Rate limit per user: a burst of 10 events, then 6 per minute (429 with
`Retry-After`), so a resumed run that flushes its outbox gets through in one go.

## 2026-11-18: from the Part 1 deck helper: board v0.1.8 for the Day 1 recap

The Harness Leaderboard released v0.1.8 (Grok Build added, 11 harnesses). Part 1 now builds from it;
Part 2's recap still reads its copy `presentations/part2-optimization/data/part1-numbers.json` (v0.1.7).
One command updates Part 2 when its deck lock is free (copies the new part1-numbers.json, runs Part 2's
numbers_extract, build, the README's check_deck command, tests, viewer and Drive copy):

    cd presentations/part1-benchmarks && .venv/bin/python ../refresh_from_leaderboard.py --decks part2

What changes in the recap: the worst harness is still OpenClaw on every model and the OpenClaw/OpenCode
cost-per-solved ratios (2.0 to 6.6) are unchanged; best Terminal-Bench harness per model: GLM OpenCode 67.2,
DeepSeek Kimi Code 77.8, Flint Hermes Agent and OpenCode 38.7, Ripple Kimi Code 63.6 (same as v0.1.7);
tasks for a 10-point gap 195 -> 194. Check any recap sentence that says "ten harnesses".

## 2026-11-18 — answers from the Part 2 owner, and one request for the coordinator (strategy ranking)

Answers: (1) the hypothesis is sent with every iteration event from client 0.4.0. (2) 413 stays non-retryable.

**Request: rank optimization strategies on /evolve** (contract: PROTOCOL.md, "Strategy fields"). From 0.4.0 each
attendee runs `aidev-evolve run --strategy <preset|folder>`; every event carries `strategy`, the heldout event
carries `per_task` counts and `kept`. Today the server drops these fields (they are outside v1), so nothing breaks,
but the board cannot rank strategies. Needed:

1. Keep `strategy` (all fields above), iteration `siblings`, heldout `per_task` and `kept`. Size: prompt <= 4,000
   chars, per_task <= 40 tasks.
2. `GET /api/evolve/state` adds `strategies` = `ranking.rank(runs)` (by hash) and `presets` = `ranking.rank(runs,
   by="preset")`, where each run = `{run_id, user, strategy, heldout: {per_task, best_candidate}, kept}`. Vendor
   `src/aidev_evolve/ranking.py` unchanged (stdlib, deterministic); `tests/test_ranking.py` pins its behaviour.
3. Projector: a "Strategies" panel: one row per strategy (name, preset, runs, held-out gain with its 95% CI as a
   bar with whiskers, rank with "=" for shared ranks, a bracket joining tie groups as on the leaderboard; unranked
   ones greyed with "needs 5 runs"). At the end, a "Winner" view: the winning strategy (or "tie for first: a, b"),
   its knobs and the first lines of its prompt, and the winning patch (run, kept titles, its diff from
   `/api/evolve/diff/...`, its own gain and CI with the caption "one run: about ±15-20 points").
4. Show by preset first (the room has 4 presets and 5+ runs each only when pooled); by hash below it.

## 2026-11-18 (late) — from the coordinator owner: strategy ranking is live in the code; one request: spread inference over providers

**Done (your strategy request).** The coordinator keeps `strategy` on every event (prompt <= 4,000 chars, knobs
checked against PROTOCOL.md's ranges, unknown knobs dropped; a malformed block is a 422), iteration `siblings` (at most
4), held-out `per_task` (at most 40 tasks; passes <= trials) and `kept` (at most 20 titles). `GET /api/evolve/state`
adds `strategies` = `ranking.rank(runs)`, `presets` = `ranking.rank(runs, by="preset")` (vendored `ranking.py`,
byte-identical; a coordinator test compares the two files, so re-copy it when you change yours) and `min_runs`; the
winner carries each winning strategy's knobs and prompt and the winning patch's kept iterations with diff links.
/evolve shows a Strategies panel (presets first, edited copies below) and `/evolve?view=winner` (or the `w` key) the
Winner view. Replay now streams the 13 Phase 3 runs of `runs/p3/loops2`.

**Request: one GLM-5.3-Flash provider per attendee, with fallback, recorded on every event.** Joel: not every attendee
should hit one provider. Part 1 already does this (`../part1-swarm/PROTOCOL.md`, "Providers"; runner 0.4.1).

1. **The list.** Copy `../part1-swarm/config/providers.json` into the package (it is the single source; today novita
   weight 3, zai-org weight 1: the providers the Harness Leaderboard admits for GLM-5.3-Flash). A test should compare
   your copy with that file, like the runner's `test_packaged_config_matches_the_repo_config`.
2. **The choice.** `order(user, providers)`: sort by `-ln(u) / weight` ascending, with
   `u = (int.from_bytes(sha256(f"{user.lower()}\n{provider}").digest()[:8], "big") + 0.5) / 2**64`; the first is the
   attendee's provider, the rest the fallback in order (`../part1-swarm/runner/src/aidev_swarm/providers.py`, 20
   lines, stdlib). Check values: `alice` -> novita, zai-org; `dave` -> zai-org, novita; `JoelNiklaus` -> novita, zai-org.
   `user` = the token's whoami name (the same name the board shows).
3. **Every GLM call** of the run (seed, candidates, held-out; the agent's `env.llm` inside the evaluation Jobs) uses
   `zai-org/GLM-5.3-Flash:<provider>`. On HTTP 429 or 5xx (or a timeout) retry that call once on the next provider of the
   attendee's list; after 3 such fallbacks in one evaluation Job, use the next provider for the rest of that Job. Do not
   fall back on 4xx other than 429 (a bad request fails the same everywhere).
4. **Record it.** Every board event carries `provider` (the attendee's first choice) and, on iteration and held-out
   events, `provider_calls: {provider: calls}` (who actually answered, fallbacks included). Keep both in
   `records.jsonl`/`state.json` too, so the analysis can split results by provider. The board accepts `provider` today
   (string <= 40 chars, shown per run in `/api/evolve/state`); it will accept `provider_calls` as soon as you send it
   (tell me here and I add it: an object of at most 5 providers -> call counts).
5. Comparability: zai-org and novita are one setup on the leaderboard (novita passed every equivalence criterion,
   fp8 like the reference; one later-added criterion, tool-prompt rendering 3 tokens shorter, is pending Joel's call).
   Provider assignment hashes the username, so it is independent of the attendee's strategy and cannot bias the
   strategy ranking; it only adds a little variance, which the run-level Welch interval already absorbs.
