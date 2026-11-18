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
| 3 | `GET /api/evolve/state` | Cached 2 s with ETag, like `/api/state`: `totals` {attendees, runs, iterations, kept, spend_usd (sum of each run's latest)}; `runs` (best first, at most 200; each `curve` [[iteration, best_val]], seed at 0, `heldout`); `room_best` (best any run reached by each iteration); `baseline` (mean seed validation score; the recorded runs' while the room is empty); `reference` (the recorded runs' curves); `feed` (newest 12 kept patches: user, iteration, title, one-line hypothesis, delta = val_child - val_parent, lines added/removed); `fixes` (top 6 fix categories by proposals, with kept counts; keyword rules on the title, then the hypothesis). `GET /api/evolve/diff/{user}/{run_id}/{iteration}` returns the stored diff as text/plain. |
| 4 | `GET /evolve` | Projector page, same style as `/`, no text under 26 px at 1920x1080: race chart (every run thin grey, top 5 coloured and named at the right edge, room best as a white band, seed baseline dashed, recorded runs dotted), patch feed of kept patches, "most common fixes" bars (tried / kept), counters (attendees, iterations, patches kept, HF credits). Polls every 10 s (`?every=3` on the projector). |
| 5 | Replay layer | `replay/evolve-recorded.json` (built by `replay/build_evolve_replay.py` from the Phase 2 runs B1, B2 in the talk folder: seed, 8 iterations, held-out each, at their recorded times). `coordinator --replay` streams them as users `recorded-B1`/`recorded-B2` (REPLAY badge, read-only); in live mode they are the race chart's dotted backdrop. |

Load test (`swarm-loadtest --evolve`, local, 18 Nov): 500 attendees posting every 30 s (seed + 6 iterations each,
4 KB diffs) plus 60 viewers polling every 3 s: 3,500 submits, 0 errors, 0 rate-limited, submit p50 4 ms / p95 13 ms,
state p50 57 ms / p95 189 ms; the board counted 500 attendees and 3,000 iterations.

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

`verdict` is one of `kept`, `dropped`, `dropped: did not hold on re-measure`, `rejected: audit`, `rejected: crashed in preflight`, `skipped: evaluation failed`,
`skipped: agent.py unchanged`, `skipped: proposer failed`. Scores are pass rates in [0, 1] on the fixed
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
