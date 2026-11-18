# Verify before finishing, show clock, harden calls
Hypothesis: runs end on the first text answer without a check (count-dataset-tokens, sparql-university FAIL with successful-looking transcripts); bn-fit-modify hit the timeout with no clock; bad tool arguments would crash the run.
Change: prompt asks for exact requirements and a final check.
Change: one verification nudge when the model first answers in text and more than 40 s remain.
Change: every tool result ends with "[N s left]".
Change: malformed tool arguments return an error message instead of raising.
Expected: fixes premature unchecked finishes and helps timeout awareness; could cost 10-20 s on tasks already right, or make the model over-edit a correct result.