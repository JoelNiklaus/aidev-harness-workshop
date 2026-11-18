"""A small terminal agent: the starting harness for aidev-evolve (Phase 4 seed E, bundled since 0.5.0; copy kept for the sweeps).

The loop: send the conversation to the model, run the bash commands it asks for, append the output,
repeat until the model answers without a tool call or the step limit is reached.

Contract (see README): `run(task, env)`; `env.llm(messages, tools=...)` returns the assistant message,
`env.shell(command, timeout=None)` runs `bash -c` in the task container and returns
(exit_code, output, timed_out), `env.time_left()` gives the seconds left, `env.log(text)` adds a note.
"""

import json

SYSTEM_PROMPT = """You are an AI agent that solves tasks in a Linux terminal.
Use the bash tool to run commands. Follow every requirement of the task exactly (file names, formats, values).
Before you finish, check your result against the task text by running it or reading the output file.
When the task is complete, reply with a short summary."""

MAX_OUTPUT_CHARS = 6000  # long output keeps its head and its tail
MAX_COMMAND_S = 150  # one command may run long (installs), but never past the budget

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
    output = result.output
    if len(output) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        output = output[:half] + f"\n[... {len(output) - 2 * half} chars cut ...]\n" + output[-half:]
    note = "\n[command timed out and was killed]" if result.timed_out else ""
    return f"exit code: {result.exit_code}\n{output}{note}"


def run_tool_call(call, env):
    try:
        command = json.loads(call["function"]["arguments"])["command"]
    except (ValueError, KeyError, TypeError):
        return "error: arguments must be JSON with a command field"
    timeout = max(5, min(MAX_COMMAND_S, int(env.time_left()) - 10))
    return format_output(env.shell(command, timeout=timeout)) + f"\n[{int(env.time_left())} s left]"


def run(task, env):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    nudged = False
    while env.time_left() > 8:  # the time budget is the only limit
        reply = env.llm(messages, tools=[BASH_TOOL])
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            if nudged or env.time_left() < 40:
                return  # the model answered in text: we are done
            nudged = True  # once: ask for a check against the task before accepting
            messages.append({"role": "user", "content": "Before finishing, re-read the task and verify each requirement with a command. Fix anything that does not match, then summarize."})
            continue
        for call in tool_calls:
            content = run_tool_call(call, env)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
