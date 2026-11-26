# Hint 2

Read only as far as you need. The symptom is often enough: go back to the trajectories and find the cause yourself.

## The symptom

Look at the column `model calls` in `INDEX.md` of the newest folder under `traces/`, next to the column
`agent time`. Several failed tasks stop at the same number of model calls, with most of their 120 seconds unused.
Their trajectories end in the middle of the work, with a command and no final answer.

A question for your coding agent: "Which failed tasks stopped while the model was still working, and what stopped them?"

## The setting

`TURN_LIMIT = 10` in `agent.py` ends every run after 10 model calls. Raise it (for example to 100): the time per
task still ends a run that goes on too long.

Then tell your coding agent the change in your own words. It checks it and makes the edit.
