# Hint 1

Read only as far as you need. The symptom is often enough: go back to the trajectories and find the cause yourself.

## The symptom

Open a failed trajectory in the newest folder under `traces/` and compare the section `## task` with the first line
of step 1. That line says how many characters of the task the model received. In several failed tasks the model
did everything it was told and still failed: the last requirements of the task never reached it.

A question for your coding agent: "Compare the task text with what the model received in step 1 of the failed tasks."

## The setting

`INSTRUCTION_CHARS = 1000` in `agent.py` cuts every task text to 1,000 characters. Set it to `None`.

Then tell your coding agent the change in your own words. It checks it and makes the edit.
