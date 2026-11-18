Read `traces/INDEX.md` and every failed transcript in `traces/`, then `HISTORY.md`, `CONTRACT.md` and `agent.py`.

Before you edit anything, write `FAILURES.md`: one line per failed transcript naming every mechanism that hurt it
(ran out of steps or stopped early, a command marked TIMED OUT, the output it needed was cut, an error it never saw,
it stopped without checking). A transcript can show several. Count each mechanism, and for each one name the line
of `agent.py` that causes it.

Then fix every mechanism that appears in two or more failed transcripts, all in this one patch: one small change
per mechanism, each fixing a cause in `agent.py`, none fixing a single task. Skip a mechanism an earlier kept patch
already fixed; do not resend a dropped change unchanged. List each fix on its own line under `Change:` in PATCH.md.
