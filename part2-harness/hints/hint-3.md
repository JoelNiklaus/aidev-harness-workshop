# Hint 3

Read only as far as you need. The symptom is often enough: go back to the trajectories and find the cause yourself.

## The symptom

Some tasks ask for a server that keeps running, for example `kv-store-grpc` and `pypi-server`. In their
trajectories the model starts the server in the background, checks it in the same command, and reports success.
The task still fails. Read the full script that the container ran for one command (the line that starts with `$`):
what happens after the model's own command has finished?

A question for your coding agent: "In the tasks that need a running server, what happens to the server after the command that started it?"

## The setting

`ISOLATE_COMMANDS = True` in `agent.py` adds a line to every command that ends all processes the command started,
so no server survives its own command. Set it to `False`.

Then tell your coding agent the change in your own words. It checks it and makes the edit.
