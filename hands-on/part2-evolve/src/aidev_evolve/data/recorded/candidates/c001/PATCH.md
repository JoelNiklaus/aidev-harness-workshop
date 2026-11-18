# Run until time budget, not 10 steps
Hypothesis: all four transcripts end "returned" after exactly 10 model calls and only 18-32 s of the 180 s budget; the model was still exploring (listing files, checking tools) when the step cap cut it off.
Change: MAX_STEPS 10 -> 100 as a safety net; the loop now stops when env.time_left() drops below a 10 s margin.
Expected: tasks needing more than 10 commands (build, recovery, multi-step data work) get time to finish; could burn time on hopeless tasks, or hit the time limit mid-command.