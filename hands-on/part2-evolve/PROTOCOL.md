# Evolve board protocol (Part 2 hands-on) — proposal, v0 (17 Nov 2026)

Goal: every attendee runs their own evolve loop (`aidev-evolve run`); the projector shows the room's
progress next to Part 1's board. This file specifies what the coordinator Space needs. It is built
**only here** for now: `src/aidev_evolve/board.py` is the client, `devtools/fake_board.py` a stdlib
stand-in with the page. The coordinator owner integrates it into `../part1-swarm/coordinator/` later.

## What the coordinator needs (to build)

| # | Addition | Notes |
|---|---|---|
| 1 | `POST /api/evolve/submit` | Auth exactly as Part 1 (Bearer HF token -> whoami-v2 username, cached; tokens never stored or logged). Body = one Event (below). Idempotent on `(user, run_id, kind, iteration)`: a retried POST replaces, never duplicates. 422 on a bad body. Rate: about 1 event per 5 min per attendee, so 500 attendees = under 2 requests/s. |
| 2 | Storage | One table `evolve_events(user, run_id, kind, iteration, received_at, json)`; the same results bucket as Part 1 under `evolve/<user>/<run_id>.jsonl`. |
| 3 | `GET /api/evolve/state` | Cached 2 s, like `/api/state`: per run the curve of `best_val` over iterations, the latest `spend_usd`, counts kept/dropped, the newest 30 iteration events (title, accepted, delta, user) and up to 200 runs. Diffs only by id (`GET /api/evolve/diff/{user}/{run_id}/{iteration}` returns the stored diff text). |
| 4 | `GET /evolve` | Projector page, same style as `/` (26 px minimum): (a) **race chart**: one line per run, x = iteration, y = best validation pass rate of that run's current agent; the seed point at x = 0; top 10 runs labelled with the username; (b) **patch feed**: newest accepted patches first: user, title, `+N points`, fixed tasks, a link that opens the diff; dropped patches greyed; (c) a counter: runs, iterations, patches kept, median gain, and room spend (sum of each run's latest `spend_usd`). |
| 5 | Replay layer | The recorded Phase 1/2 runs (`src/aidev_evolve/data/recorded/`) loaded as reference lines, so the page is never empty (offline fallback, as Part 1's `replay/`). |

No spending cap is enforced here (Joel, 17 Nov): spend is submitted for display only. The design target
is that one full run (seed + about 8-10 iterations + held-out check) costs well under $3 of HF credits.

## Event (JSON body)

Every event carries `schema: "aidev-evolve/1"`, `run_id` (the client's `<user>-<run folder>`), `kind`,
and `spend_usd` (HF credits this run has used so far: inference + sandbox estimate + Job time).

```
kind = "seed":      {val_score, search_score}
kind = "iteration": {iteration, candidate, parent, title, accepted: bool, verdict, val_child|null,
                     val_parent|null, best_val, fixed: [task], broke: [task], diff: str (<= 8,000 chars)}
kind = "heldout":   {seed, best, best_candidate, n_tasks, trials}
```

`verdict` is one of `kept`, `dropped`, `dropped: did not hold on re-measure`, `rejected: audit`, `rejected: crashed in preflight`, `skipped: evaluation failed`,
`skipped: agent.py unchanged`, `skipped: proposer failed`. Scores are pass rates in [0, 1] on the fixed
validation split (`src/aidev_evolve/data/splits.json`), so every attendee's curve is on the same axis.

## Client behaviour (built)

- `aidev-evolve run --board <coordinator URL>` posts the events; without `--board` nothing is sent.
- A failed POST prints one dim line and the loop goes on (best effort, 10 s timeout, no retry queue yet).
