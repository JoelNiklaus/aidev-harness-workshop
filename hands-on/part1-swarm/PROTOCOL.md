# Swarm evaluation protocol (Part 1 hands-on)

Goal: hundreds of attendees' coding agents fill a grid of Terminal-Bench 2.1 trials together; a live board shows results as they land. Everything runs on Hugging Face infrastructure.

## Components
- `coordinator/` — HF Space (Docker, FastAPI, one process). Owns the queue and the board.
- `runner/` — attendee CLI (`aidev-swarm`), run with `uvx` from a small package. Claims a cell, runs it with Harbor in an HF Sandbox under the attendee's own HF token, submits the result.
- `replay/` — pre-run results from existing Harness Leaderboard trials, loaded as the offline fallback and as the "reference" layer.

## Grid
`config/grid.json`: `{"model": "<HF router id of GLM-5.3-Flash>", "harnesses": [...], "tasks": [...], "trials_per_cell": K}`. A cell = (harness, task, trial_index). Task list comes from `config/tb21-subset.json`.

## HTTP API (coordinator)
Auth: every POST carries `Authorization: Bearer <attendee HF token>`; the coordinator calls `https://huggingface.co/api/whoami-v2` once per token (cache it) and records the username. Never store or log tokens.

- `POST /api/claim` → `{"cell_id", "harness", "task", "trial", "model", "lease_expires_at", "run_config": {...}}` or `204` when nothing is left. Atomic (single process + SQLite). Lease 40 min; expired leases return to the queue. Harness order in `grid.json`: Pi, OpenCode, OpenClaw first; then the remaining 7 sorted by expected trial duration/cost, longest and most expensive first (they need the most time to finish). Order: one harness after the next, in the order of `grid.json` `harnesses` — all cells of harness 1 are handed out before any cell of harness 2. Within a harness, breadth-first over trials (every task gets trial 0 before any gets trial 1). The grid holds 10 harnesses; the room may not reach all of them, and the board shows only harnesses with results (plus "not reached" for the rest). One active lease per user unless `?n=2`.
- `POST /api/heartbeat/{cell_id}` → extends the lease (runner calls every 2 min).
- `POST /api/submit/{cell_id}` body = Result (below) → `200`. Rejects if the lease belongs to another user. Writes the result JSON to the results bucket/dataset and to SQLite.
- `POST /api/release/{cell_id}` → gives a cell back (runner on crash / Ctrl-C).
- `GET /api/state` → grid fill counts, per-(harness) aggregates, recent results (for the board, polled every 3–5 s; cache 2 s).
- `GET /` → live board: (1) leaderboard table per harness: pass rate with CI, trials, cost per trial, cost per pass, tokens per trial; (2) grid fill view (harness × task, cells colored claimed/pass/fail); (3) Pareto plot cost per trial (log x) vs pass rate, room results next to the reference layer; (4) feed of the newest results with usernames. Readable on a projector: large type.
- `GET /join` → the copy-paste page the slide's QR code opens: install uv, token permissions, `uvx hf auth login`, the agent prompt and the launch command, each with a Copy button; a link back to the board. Built from `runner/README.md` (wheel URL) and `runner/AGENT_PROMPT.md`; no auth header needed.

## Result (JSON)
```
{"cell_id": str, "harness": str, "harness_version": str, "task": str, "trial": int, "model": str,
 "reward": float, "passed": bool, "started_at": iso, "finished_at": iso, "wall_s": float,
 "n_input_tokens": int, "n_cached_tokens": int|null, "n_output_tokens": int, "n_calls": int|null,
 "cost_usd": float, "error": str|null, "hf_user": str, "runner_version": str,
 "sandbox_id": str|null, "trajectory_url": str|null}
```
`cost_usd` = tokens × the provider list price in `config/prices.json` (same numbers on board and slides).

## Runner UX
`uvx --from <wheel url> aidev-swarm check` tests an account before the workshop (one model call, one tiny HF Job, one sandbox command; claims nothing). `aidev-swarm launch [--n K]` submits one HF Job in the attendee's namespace that runs `aidev-swarm run --loop | --n K`: it claims, runs the trial, heartbeats, submits and prints the board URL. `run --dry` checks token and credits and claims nothing. A one-paragraph prompt for the attendee's coding agent ("run this, watch it, report the result") lives in `runner/AGENT_PROMPT.md`.

## Rules
A claim beyond the user's lease limit returns 409 with `held` (cells already held); `?n=2` returns a list of up to two cells; cell ids are `<harness>__<task>__<trial>`; bad requests 422; Hub outage during auth 503. The HF edge rate-limits per IP (429 after ~870 fast requests): runners keep unsent results and retry for at least 30 min; the board polls every 10 s with backoff.

Development: every test fakes the external calls (Hub, Jobs, sandboxes, model) and the suites run in seconds. Tokens never appear in logs.
