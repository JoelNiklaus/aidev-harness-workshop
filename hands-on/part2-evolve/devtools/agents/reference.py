"""Reference harness (development only): the seed with every known weakness fixed by hand.

Used to find tasks where the harness, not the model, decides the outcome (seed fails, reference passes
at the same time budget). Never shown to the proposer.
"""

import json

SYSTEM_PROMPT = """You are an AI agent that solves tasks in a Linux terminal.
Use the bash tool to run commands. Each command runs in a fresh shell (use absolute paths or `cd dir && ...`).
Start servers and other long-running programs in the background (`nohup cmd > /tmp/log 2>&1 &`).
Work quickly: you have limited time. Before you finish, verify the result against every requirement of
the task (run it, check exact file names, paths and formats). When everything is verified, reply with a
short summary and no tool call."""

COMMAND_TIMEOUT_S = 60
HEAD_CHARS, TAIL_CHARS = 2000, 4000
VERIFY_NUDGE = ("Before you stop: re-read the task and check each requirement with commands (run the code or tests, "
                "check exact paths, names and output formats). Fix anything missing. If all is verified, reply DONE.")

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its output (stdout and stderr). Commands time out after "
                       f"{COMMAND_TIMEOUT_S} s unless you pass a larger `timeout` in seconds.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."},
                           "timeout": {"type": "integer", "description": "Seconds before the command is killed."}},
            "required": ["command"],
        },
    },
}


def format_output(result, env):
    out = result.output or ""
    if len(out) > HEAD_CHARS + TAIL_CHARS:
        out = f"{out[:HEAD_CHARS]}\n[... {len(out) - HEAD_CHARS - TAIL_CHARS} chars cut ...]\n{out[-TAIL_CHARS:]}"
    note = f"\n[command timed out: killed. Run long jobs in the background or pass a larger timeout]" if result.timed_out else ""
    left = env.time_left()
    warn = " Time is almost up: write your final answer files now." if left < 30 else ""
    return f"exit code: {result.exit_code}{note}\n{out}\n[{left:.0f} s left]{warn}"


def run_tool_call(call, env):
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
        command = args["command"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"error: could not read the tool call arguments ({e}); send JSON like {{\"command\": \"ls\"}}"
    timeout = args.get("timeout") or COMMAND_TIMEOUT_S
    result = env.shell(command, timeout=min(float(timeout), max(5.0, env.time_left() - 5)))
    return format_output(result, env)


def run(task, env):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    nudged = False
    while True:
        reply = env.llm(messages, tools=[BASH_TOOL])
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            if nudged or env.time_left() < 20:
                return
            nudged = True
            messages.append({"role": "user", "content": VERIFY_NUDGE})
            continue
        for call in tool_calls:
            content = run_tool_call(call, env)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
