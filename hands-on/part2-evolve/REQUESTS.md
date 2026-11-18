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
