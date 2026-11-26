"""A small terminal agent.

One model, one tool. Each turn sends the conversation to the model, runs the bash commands it asks
for in the task container, and feeds the results back. The run ends when the model answers in plain
text.

Contract (see CONTRACT.md): `run(task, env)` is called once per task.
  env.llm(messages, tools=...)       -> the assistant message (OpenAI chat format)
  env.shell(command, timeout=None)   -> result with exit_code, output, timed_out
  env.time_left()                    -> seconds left of the task budget
  env.log(text)                      -> adds a note to the transcript

Layout of this file:
  settings            the knobs, all in one place
  output handling     clip, format_result
  prompt building     build_messages
  history handling    visible_history
  model call          sampling_params, ask_model
  command execution   parse_command, wrap_command, run_command, answer_tool_call
  stop rule           is_final
  main loop           run
"""

import json

# --------------------------------------------------------------------------- settings

SYSTEM_PROMPT = """You are an AI agent that solves tasks in a Linux terminal.
Use the bash tool to run commands. When the task is complete, reply with a short summary."""

# Model
TEMPERATURE = None          # None = provider default
REPLY_TOKENS = None         # None = provider default

# Conversation
TURN_LIMIT = 10             # model calls per task
INSTRUCTION_CHARS = 1000    # size of the task message
HISTORY_MESSAGES = None     # None = send the whole conversation every turn

# Commands
COMMAND_TIMEOUT_S = 10      # seconds per command
ISOLATE_COMMANDS = True     # every command starts from a clean process table
OUTPUT_STREAMS = "stdout"   # "stdout" or "both"

# Tool results
OUTPUT_CHARS = 1000         # size of one tool result
SHOW_EXIT_CODE = True

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

# Shell epilogue for isolated commands: keep the command's exit status and leave no children behind.
_REAP = '; rc=$?; for p in /proc/[0-9]*; do p=${p#/proc/}; [ "$p" -gt $$ ] && kill -9 "$p" 2>/dev/null; done; exit $rc'


# --------------------------------------------------------------------------- output handling

def clip(text, limit):
    """Keep `text` within `limit` characters (None = no limit)."""
    if limit is None:
        return text
    return text[:limit]


def format_result(result):
    """What the model sees of a command's result."""
    body = clip(result.output, OUTPUT_CHARS)
    if SHOW_EXIT_CODE:
        return f"exit code: {result.exit_code}\n{body}"
    return body


# --------------------------------------------------------------------------- prompt building

def build_messages(task):
    """The conversation the first turn starts from."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clip(task, INSTRUCTION_CHARS)},
    ]


# --------------------------------------------------------------------------- history handling

def visible_history(messages):
    """The part of the conversation that is sent to the model this turn."""
    if HISTORY_MESSAGES is None or len(messages) <= HISTORY_MESSAGES:
        return messages
    opening = messages[:2]  # system prompt and task
    recent = messages[-(HISTORY_MESSAGES - len(opening)):]
    while recent and recent[0]["role"] == "tool":  # never start on a tool result without its call
        recent = recent[1:]
    return opening + recent


# --------------------------------------------------------------------------- model call

def sampling_params():
    params = {}
    if TEMPERATURE is not None:
        params["temperature"] = TEMPERATURE
    if REPLY_TOKENS is not None:
        params["max_tokens"] = REPLY_TOKENS
    return params


def ask_model(messages, env):
    """One model call. Returns the assistant message."""
    return env.llm(visible_history(messages), tools=[BASH_TOOL], **sampling_params())


# --------------------------------------------------------------------------- command execution

def parse_command(call):
    """The bash command of one tool call."""
    args = json.loads(call["function"]["arguments"])
    return args["command"]


def wrap_command(command):
    """Turn the model's command into the script the container runs.

    The command runs as one group, so a multi-line command gives one exit status and one output.
    """
    script = "{ " + command + "\n}"
    if OUTPUT_STREAMS == "stdout":
        script += " 2>/dev/null"
    if ISOLATE_COMMANDS:
        script += _REAP
    return script


def run_command(command, env):
    """Run one command in the task container and return the text for the model."""
    result = env.shell(wrap_command(command), timeout=COMMAND_TIMEOUT_S)
    return format_result(result)


def answer_tool_call(call, env):
    """The tool message that answers one tool call."""
    content = run_command(parse_command(call), env)
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


# --------------------------------------------------------------------------- stop rule

def is_final(reply):
    """A reply without tool calls is the model's answer."""
    return not (reply.get("tool_calls") or [])


# --------------------------------------------------------------------------- main loop

def run(task, env):
    messages = build_messages(task)
    for turn in range(TURN_LIMIT):
        reply = ask_model(messages, env)
        messages.append(reply)
        if is_final(reply):
            return
        for call in reply["tool_calls"]:
            messages.append(answer_tool_call(call, env))
