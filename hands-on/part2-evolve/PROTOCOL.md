# Evolve board protocol (Part 2 hands-on) — proposal, v0 (17 Nov 2026)

Goal: every attendee runs their own evolve loop (`aidev-evolve run`); the projector shows the room's
progress next to Part 1's board. This file is the contract between the client and the coordinator Space. Here,
`src/aidev_evolve/board.py` is the client and `devtools/fake_board.py` a stdlib
stand-in with the page. The real server runs in the coordinator Space (`../part1-swarm/coordinator/`, section below).

## Coordinator side (built 18 Nov in `../part1-swarm/coordinator/`; this section is owned by the coordinator helper)

| # | Endpoint / part | As built |
|---|---|---|
| 1 | `POST /api/evolve/submit` | Auth exactly as Part 1 (Bearer HF token -> whoami-v2 username, cached; tokens never stored or logged); the username comes from the token, never from `run_id`. Body = one Event (below). Idempotent on `(user, run_id, kind, iteration)`: a retry replaces the event and keeps its first arrival time. Fields outside the protocol are dropped; overlong text is cut (title 200, hypothesis 1,000, diff 8,000, task names 100, at most 100 fixed/broke tasks), not refused. 422: wrong type or range, unknown `kind`, bad `run_id` (1-100 of `[A-Za-z0-9._-]`), more than 10 runs per user. 413: body over 64 KB (checked before auth). 429 + `Retry-After`: more than a burst of 10, then 6 per minute, per user. 503 in replay mode and on a Hub outage. |
| 2 | Storage | SQLite `evolve_events(user, run_id, kind, iteration, received_at, json, diff)`; the Part 1 results bucket under `evolve/<user>/<run_id>/<kind>-<iteration>.json` (one small file per event, flushed every 5 s with Part 1's batch); a restarted Space rebuilds from it. |
| 3 | `GET /api/evolve/state` | Cached 2 s with ETag, like `/api/state`: `totals` {attendees, runs, iterations, kept, spend_usd (sum of each run's latest)}; `runs` (highest `current` first, at most 200; each `curve` is a step line of the validation pass rate of the run's current harness: one level per kept patch (the seed first), each at that harness's latest pooled `best_val`, so it moves only at a kept patch; `current` its last level; `heldout`); `leader` (the highest current harness at each iteration, the same kind of step line); `baseline` (mean seed validation score; the recorded runs' while no run has started); `reference` (the recorded runs' curves, built the same way); `feed` (newest 12 kept patches: user, iteration, title, one-line hypothesis, delta = val_child - val_parent, lines added/removed); `fixes` (top 6 fix categories by proposals, with kept counts; keyword rules on the title, then the hypothesis). `GET /api/evolve/diff/{user}/{run_id}/{iteration}` returns the stored diff as text/plain. |
| 4 | `GET /evolve` | Projector page, same style as `/`, no text under 26 px at 1920x1080: race chart "Validation pass rate of each run's current harness" (every run thin grey, top 5 coloured and named at the right edge, the leading run as a white band, seed baseline dashed, recorded runs dotted), patch feed of kept patches (title, then the hypothesis cut to at most two lines at a sentence or clause boundary), "most common fixes" bars (tried / kept), counters (attendees, iterations, patches kept, HF credits). Polls every 10 s (`?every=3` on the projector). |
| 5 | Replay layer | `replay/evolve-recorded.json` (built by `replay/build_evolve_replay.py` from the Phase 4 runs in the talk folder's `evolve/runs/p4/loops`: 20 runs, baseline 8, every-weakness 7, code-audit 5, seed E, each with its strategy block, seed, 2-4 iterations and held-out with per-task counts over 21 tasks, at their recorded times; code-audit ranks first alone; `provider`/`provider_calls` are copied when the run recorded them). `coordinator --replay` streams them under their run names (REPLAY badge, read-only); in live mode they are the race chart's dotted backdrop. |
| 6 | Strategy ranking (client 0.4.0 fields) | Kept: `strategy` on every event (name <= 60, preset, hash, modified, prompt <= 4,000, knobs in PROTOCOL ranges; 422 when malformed), `siblings` (<= 4), heldout `per_task` (<= 40 tasks, passes <= trials) and `kept` (<= 20), `provider` (<= 40), and (client 0.5.0) `provider_calls` on iteration and held-out events (object, at most 5 providers kept, each name <= 40, integer counts 0-1,000,000; 422 otherwise). Any preset name of `[a-z0-9-]` up to 40 chars is accepted, so `every-weakness` and `code-audit` need no server change. The prompt is stored with seed and held-out events only. `/api/evolve/state` adds `strategies` (`ranking.rank(runs)`, by hash), `presets` (`by="preset"`), `min_runs`; each `winner` adds `strategies` (knobs, prompt, gain, CI; a preset pooled with edited copies shows the preset's own) and `patch.iterations` (kept iterations with `diff_url`). `ranking.py` is vendored byte-identical (a test compares the files). Ranking 500 runs costs ~2 s here, so it reruns only when a held-out result changed and then at most every 10x its last cost (`RankCache`). Runs carry `strategy`, `preset`, `provider` and `provider_calls` (summed over the run's iteration and held-out events). |
| 7 | Strategies panel and Winner view | /evolve right column: "Strategies" above the patch feed: by preset first, edited copies below; per row rank ("=" for shared ranks), name, runs, held-out gain as a dot with its 95% Welch interval as whiskers (axis set by strategies with 3+ runs, wider whiskers clipped), the gain in points; tie groups joined by a bracket; unranked rows greyed, "needs 5 runs". `/evolve?view=winner` (or key `w`): the winning strategy or "Tie for first", its knobs and the first 8 lines of its prompt; the winning patch: run, kept titles, its diff, its own gain and CI with "one run: about ±15-20 points". All text >= 26 px at 1920x1080. |

Load test (`swarm-loadtest --evolve`, local, 18 Nov): 500 attendees posting every 30 s (seed + 6 iterations each,
4 KB diffs) plus 60 viewers polling every 3 s: 3,500 submits, 0 errors, 0 rate-limited, submit p50 4 ms / p95 13 ms,
state p50 57 ms / p95 189 ms; the board counted 500 attendees and 3,000 iterations. With strategies (every event
carries a strategy block, plus a held-out event with 23 tasks' counts; 500 attendees every 15 s, 60 viewers): 4,000
submits, 0 errors, 0 rate-limited, submit p50 4 ms / p95 70 ms (max 2.4 s while a 500-run ranking is computed),
state p50 57 ms / p95 700 ms; all 500 runs ranked in 4 presets.

No spending cap is enforced here (Joel, 17 Nov): spend is submitted for display only. The design target
is that one full run (seed + about 8-10 iterations + held-out check) costs well under $3 of HF credits.

## Event (JSON body)

Every event carries `schema: "aidev-evolve/1"`, `run_id` (the client's `<user>-<run folder>`), `kind`,
and `spend_usd` (HF credits this run has used so far: inference + sandbox estimate + Job time).

```
kind = "seed":      {val_score, search_score}
kind = "iteration": {iteration, candidate, parent, title, accepted: bool, verdict, val_child|null,
                     val_parent|null, best_val, fixed: [task], broke: [task], diff: str (<= 8,000 chars),
                     hypothesis: str|null (optional; requested in REQUESTS.md, shown in the feed)}
kind = "heldout":   {seed, best, best_candidate, n_tasks, trials}
```

`provider` (every kind, optional, requested 18 Nov in REQUESTS.md): the GLM-5.3-Flash provider the attendee's run uses,
picked per attendee from `../part1-swarm/config/providers.json` exactly as Part 1 does (`../part1-swarm/PROTOCOL.md`,
"Providers": weighted rendezvous hash of the lowercased HF username; on 429/5xx a call falls back to the next provider
of the attendee's list). Requested next to it: `provider_calls: {provider: calls}` on iteration and held-out events.

### Strategy fields (added 18 Nov, client 0.4.0; additive: the schema string stays `aidev-evolve/1`)

Every event (seed, iteration, heldout) also carries the run's optimization strategy:

```
strategy = {name: str (<= 60), preset: str (one of baseline, failures-first, small-diffs, two-candidates, every-weakness,
            code-audit; the last two since client 0.5.0),
            hash: str (10 hex: knobs + prompt; identical strategies share it across attendees),
            modified: bool (the attendee edited the preset), prompt: str (<= 4,000 chars, the proposer approach),
            knobs: {traces: "failed"|"all", max_traces: 1-8, history: "none"|"titles"|"full", diffs: bool,
                    candidates: 1|2, val_trials: 1-3, z: 0-3, confirm: bool, iterations: 1-6}}
```

`kind = "iteration"` adds `siblings: [{candidate, title, val: float|null, verdict}]` (the second proposal of a
two-candidate iteration: `dropped: lost the screen`, or its audit/preflight verdict; empty otherwise) and
`hypothesis` (see REQUESTS.md). `kind = "heldout"` adds:

```
per_task: {task: [seed_passes, seed_trials, final_passes, final_trials]}   (23 tasks x 2 trials in config C)
kept:     [title of each kept patch, in order]
diff, ci_low, ci_high: the run's own paired gain and its 95% task-bootstrap CI (shown, never ranked)
```

### Provider fields (added 19 Nov, client 0.5.0; additive)

Every event carries `provider` (str: the attendee's GLM-5.3-Flash provider, the first of
`providers.order(user, data/providers.json)`, the same weighted rendezvous hash and list as Part 1). `kind = "iteration"`
and `kind = "heldout"` add `provider_calls: {provider: calls}` (who answered that iteration's or the held-out check's
model calls, fallbacks included; at most 5 providers). Fallback inside a Job: a call answered 429/5xx or timed out is
retried once at once on the next provider; after 3 such fallbacks the rest of the Job starts on the next provider.
`records.jsonl` keeps `provider_calls` per trial and `config.json` the attendee's order (`providers`).

Ranking (reference implementation `src/aidev_evolve/ranking.py`, stdlib only, deterministic: vendor it unchanged,
or match its output on `tests/test_ranking.py`): pool runs per `strategy.hash` (a view by `preset` pools edited copies
with their preset); a run's score = mean per-task final pass rate on the held-out tasks; gain = score minus the seed
pooled per task over every run in the room; CI of a strategy = Welch t interval over its run scores; two strategies
tie when the Welch interval of their difference covers 0; tie groups top-down with the Harness Leaderboard rule
(`tie_groups`, ranks shared); strategies with fewer than 5 runs are listed, not ranked. Final view: the winning
strategy (group 1; "tie for first" when it holds several) with its knobs and prompt, and the winning patch = the run
with the best final score in that group, its kept patch titles, its own gain and CI, and the note that one run's
interval is about ±15-20 points. Noise math: talk folder `evolve/strategy-noise.md`.

`verdict` is one of `kept`, `dropped`, `dropped: did not hold on re-measure`, `rejected: audit`, `rejected: crashed in preflight`, `skipped: evaluation failed`,
`skipped: agent.py unchanged`, `skipped: proposer failed` (a sibling can also be `dropped: lost the screen`). Scores are pass rates in [0, 1] on the fixed
validation split (`src/aidev_evolve/data/splits.json`), so every attendee's curve is on the same axis.

## Client behaviour (built)

- `aidev-evolve run` posts the events to the coordinator (`board.ROOM_URL`) by default; `--board <url>` points
  elsewhere, `--no-board` sends nothing. It prints `<board>/evolve` first.
- Every event goes to `<run folder>/outbox.jsonl` first; the client then delivers the outbox in order (10 s timeout
  per POST). A network error, 5xx, 401, 408 or 429 leaves the rest queued for the next event; another 4xx drops
  that one event. At the end of the run it retries 4 times (5, 10, 15 s apart); `run --resume` sends whatever is
  left. The server must therefore stay idempotent on `(user, run_id, kind, iteration)`, and may receive events
  minutes late, in a burst.
- `aidev-evolve check` calls `GET /api/evolve/state` once (200 = reachable; anything else is a warning only).
