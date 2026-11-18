"""Static checks on a proposed agent.py before any credits are spent on it.

agent.py runs inside the attendee's HF Job, next to their token. These rules keep a candidate general
(no benchmark task names) and inside its contract (only `env` reaches the model and the sandbox): imports
from an allowlist of harmless standard modules, no `open`/`eval`/`exec`/`__import__`, no private
attributes (`env._llm`). It is a guard against accidents and shortcuts, not a security sandbox.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

ALLOWED_MODULES = {"json", "re", "time", "textwrap", "shlex", "collections", "dataclasses", "itertools", "functools", "math",
                   "random", "string", "typing", "base64", "hashlib", "difflib", "enum", "datetime", "__future__"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "globals", "vars", "breakpoint", "input", "setattr", "delattr"}


def problems(source: str, task_names: Iterable[str] = ()) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]
    out: list[str] = []
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" and len(n.args.args) >= 2 for n in tree.body):
        out.append("agent.py must define run(task, env) at the top level")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"import {a.name} is not allowed (allowed: {', '.join(sorted(ALLOWED_MODULES - {'__future__'}))})"
                    for a in node.names if a.name.split(".")[0] not in ALLOWED_MODULES]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                out.append(f"import from {node.module} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            out.append(f"line {node.lineno}: {node.func.id}() is not allowed (use env.shell for the task container)")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr") and len(node.args) > 1
              and isinstance(node.args[1], ast.Constant) and str(node.args[1].value).startswith("_")):
            out.append(f"line {node.lineno}: private attribute {node.args[1].value} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            out.append(f"line {node.lineno}: private attribute .{node.attr} is not allowed")
    for name in task_names:
        if name in source:
            out.append(f"names a benchmark task ({name}): harness changes must be general")
    return out
