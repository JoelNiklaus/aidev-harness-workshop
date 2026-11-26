# Sample questions for your coding agent (Part 2)

In this folder your coding agent is your teacher. It shows you where to look, and it types the changes you decide.
It does not hand you the answer. These questions get you going. Ask them in your own words.

## Get an overview

- How does this harness work, in five lines?
- What does INDEX.md tell me, and which column do I read first?
- Which tasks ended with time left?
- Which failed tasks ended in the same way?

## Read a trajectory

- Show me step 1 of a failed task.
- Compare a passed and a failed trajectory. What is different?
- What did the task ask for, and what did the model do in the end?
- What did the last step of this trajectory do, and why did the run end there?

## Ask about the code

- What does this setting at the top of agent.py do?
- Where in agent.py does this line of the trajectory come from?

## When you are stuck

- Give me a hint.
- Give me a clearer hint.

Each time you ask, the hint gets clearer. The last resort are the files `hint-1.md` to `hint-3.md` in this folder.

## When you have an idea

Say what you saw and what you want changed: "In step 1 of two failed tasks I see this. I think that setting causes
it. Change it to this value." Your coding agent checks your idea against the trajectories and makes the change.
Then run `aidev-evolve eval` and look at the chart.
