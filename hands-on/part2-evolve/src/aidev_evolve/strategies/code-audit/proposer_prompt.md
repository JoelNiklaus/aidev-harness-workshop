Start with `agent.py` and `CONTRACT.md`, not with the transcripts. Read agent.py line by line and list in `AUDIT.md`
every line that limits, hides, cuts or changes something (a step count, a timeout, an output cap, a redirect, a cut
of the task text, anything that runs before or after the model's command). For each, write what it does to a run.

Then read `traces/INDEX.md` and the transcripts in `traces/`, failures first, and add to each AUDIT.md line the
evidence that it hurts: a transcript where it cut the run short, hid something, or undid work the model had checked
(a FAIL whose transcript looks successful points at a line that acts after the model). Read `HISTORY.md` last.

Then fix every audited line that can make a correct run fail, all in this one patch, each with the smallest change
that removes the harm. Leave lines that help alone. Skip what an earlier kept patch already fixed; do not resend a
dropped change unchanged. List each fix on its own line under `Change:` in PATCH.md.
