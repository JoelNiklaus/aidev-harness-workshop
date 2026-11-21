"""A toy agent with its own tasks: the starting harness of examples/own_eval (see APPLY-TO-YOUR-AGENT.md).

Same contract as the workshop seed: `run(task, env)`, `env.llm(messages, tools=...)` and `env.shell(command, timeout=...)`.
"""

import json

SYSTEM_PROMPT = "You solve small tasks in a Linux shell. Use the bash tool. When the task is done, reply with one short sentence."
MAX_STEPS = 3
MAX_OUTPUT_CHARS = 120
COMMAND_TIMEOUT_S = 1

BASH_TOOL = {"type": "function", "function": {"name": "bash", "description": "Run a bash command and return its output.",
                                              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}


def run(task, env):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    for step in range(MAX_STEPS):
        reply = env.llm(messages, tools=[BASH_TOOL])
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return
        for call in tool_calls:
            command = json.loads(call["function"]["arguments"])["command"]
            result = env.shell(command, timeout=COMMAND_TIMEOUT_S)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.output[:MAX_OUTPUT_CHARS]})
