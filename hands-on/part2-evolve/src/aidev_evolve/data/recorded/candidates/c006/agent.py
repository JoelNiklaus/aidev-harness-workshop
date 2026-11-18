"""A small terminal agent: the starting harness for aidev-evolve.

The loop: send the conversation to the model, run the bash commands it asks for, append the output,
repeat until the model answers without a tool call or the step limit is reached.

Contract (see README): `run(task, env)`; `env.llm(messages, tools=...)` returns the assistant message,
`env.shell(command, timeout=None)` runs `bash -c` in the task container and returns
(exit_code, output, timed_out), `env.time_left()` gives the seconds left, `env.log(text)` adds a note.
"""

import json

SYSTEM_PROMPT = """You are an AI agent that solves tasks in a Linux terminal.
Use the bash tool to run commands. When the task is complete, reply with a short summary.

Working rules:
- Time is short (about 3 minutes). First produce a complete, working deliverable in the exact
  places and formats the task names, even if rough. Only then improve it.
- If an approach fails twice, or one item eats more than about 5 commands, stop and switch to a
  different approach, or make a best guess for that item and move on to the rest.
- Never leave required outputs missing: a best-guess output beats none.
- Before you finish, check the result against the task's exact wording."""

MAX_STEPS = 100  # a safety net: the time budget is the real limit
STOP_MARGIN = 10  # seconds kept free so the last command is not cut off
MAX_OUTPUT_CHARS = 1000

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def format_output(result):
    """What the model sees of a command's result."""
    output = result.output[:MAX_OUTPUT_CHARS]
    return f"exit code: {result.exit_code}\n{output}"


def run_tool_call(call, env):
    args = json.loads(call["function"]["arguments"])
    result = env.shell("{ " + args["command"] + "\n} 2>/dev/null")  # only stdout: stderr is mostly noise
    return format_output(result)


def run(task, env):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(MAX_STEPS):
        if env.time_left() < STOP_MARGIN:
            return
        reply = env.llm(messages, tools=[BASH_TOOL])
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return  # the model answered in text: we are done
        for call in tool_calls:
            content = run_tool_call(call, env)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
