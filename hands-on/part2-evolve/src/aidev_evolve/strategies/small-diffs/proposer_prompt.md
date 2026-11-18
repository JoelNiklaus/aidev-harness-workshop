Read, in this order:
1. `traces/INDEX.md`, then the transcripts in `traces/` (the current agent.py on the search tasks).
2. `HISTORY.md`: what earlier iterations tried, and whether validation kept or dropped it. Do not repeat a
   dropped idea unchanged.
3. `CONTRACT.md` (what `env` offers) and `agent.py`.

Then make the smallest change that removes one failure mechanism you can see in the transcripts: change at
most 10 lines of `agent.py` (a limit, a timeout, one sentence of the system prompt, one condition). Small
changes are easy to attribute and rarely break passing tasks. `HISTORY.md` shows the diffs of earlier
attempts: do not resend a dropped diff.
