# The contract between agent.py and the evaluation

`run(task, env)` is called once per task, with the task text. `env` offers:

- `env.llm(messages, tools=None, *, temperature=None, max_tokens=None, top_p=None)` -> the assistant message
  as a dict: `role`, `content` (str or None) and, when the model calls tools, `tool_calls`
  (`[{"id", "type": "function", "function": {"name", "arguments": <JSON string>}}]`). OpenAI chat format:
  answer each tool call with `{"role": "tool", "tool_call_id": <id>, "content": <str>}`.
  The model (GLM-5.3-Flash), provider and reasoning effort are fixed.
- `env.shell(command, timeout=None)` -> object with `exit_code`, `output` (stdout and stderr together) and
  `timed_out`. Each call is a fresh `bash -c` in the task container (the working directory and variables do
  not carry over). The sandbox kills the command's whole process group at its timeout; without a timeout
  a command may run until the task's time budget is gone, which ends the run.
- `env.time_left()` -> seconds left of the budget. `env.log(text)` adds a note to the trajectory.

When `run` returns (or the budget is gone, or it raises), the task's hidden tests check the container.

## Fixed by the workshop

These are part of the setup, not flaws to fix:

- the model, its provider and its reasoning effort;
- the tasks and their hidden tests;
- the time budget per task (`env.time_left()` starts at 120 seconds);
- `COMMAND_TIMEOUT_S = 10` in agent.py: one command gets 10 seconds, so one evaluation stays short.
