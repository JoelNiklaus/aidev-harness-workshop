"""What the proposer reads: trajectories rendered as short markdown transcripts, failures first.

Each step shows what the model said and asked for, the command, its exit code, duration and output
(head and tail), how many characters the command printed and how many the harness passed on to the model
(so output handling is visible), and how the run ended. Long runs keep their first and last steps.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

OUT_SNIP = 700


def _snip(text: str | None, n: int = OUT_SNIP) -> str:
    text = text or ""
    if len(text) <= 2 * n:
        return text
    return f"{text[:n]}\n[... {len(text) - 2 * n:,} chars cut in this digest ...]\n{text[-n:]}"


def _task_note(s: dict[str, Any], task: str | None) -> str | None:
    """The first model call: say so when the model did not receive the task as given (cut or rewritten)."""
    users = [m.get("content") or "" for m in s.get("new_messages", []) if m.get("role") == "user"]
    if not task or not users or any(task in u for u in users):
        return None
    got = max(users, key=len)
    return f"(the model received {len(got):,} of the task's {len(task):,} chars: the harness cut or rewrote the task)"


def _llm_block(i: int, s: dict[str, Any], task: str | None = None) -> str:
    lines = [f"### step {i}: model ({s.get('duration_s', 0):.1f} s, {s.get('n_messages', '?')} messages sent)"]
    note = _task_note(s, task) if i == 1 else None
    if note:
        lines.append(note)
    seen = [m for m in s.get("new_messages", []) if m.get("role") == "tool"]
    if seen:
        lines.append("tool results the model received: " + ", ".join(f"model saw {len(m.get('content') or ''):,} chars" for m in seen))
    if s.get("dropped_or_changed_from") is not None:
        lines.append(f"(the harness changed or dropped earlier messages from message {s['dropped_or_changed_from']})")
    reply = s.get("reply") or {}
    if reply.get("content"):
        lines.append("model: " + _snip(reply["content"], 600))
    for c in reply.get("tool_calls") or []:
        lines.append(f"tool call {c.get('function', {}).get('name')}: {_snip(c.get('function', {}).get('arguments'), 400)}")
    if not reply.get("tool_calls"):
        lines.append("(no tool call)")
    return "\n".join(lines)


def _shell_block(s: dict[str, Any]) -> str:
    head = f"$ {s.get('command')}\n-> exit {s.get('exit_code')}, {s.get('duration_s', 0):.1f} s, printed {s.get('output_chars', 0):,} chars"
    if s.get("timed_out"):
        head += f", TIMED OUT (timeout {s.get('timeout')})"
    return head + "\n```\n" + _snip(s.get("output")) + "\n```"


def render(rec: dict[str, Any], max_chars: int = 20_000) -> str:
    t = rec.get("trajectory") or {}
    steps = t.get("steps", [])
    u = t.get("usage") or {}
    head = [f"# {rec['task']} — {rec['outcome'].upper()} (candidate {rec['candidate']})",
            f"ended: {t.get('end', 'no trajectory')}; agent time {t.get('agent_s', 0):.0f} s of {t.get('budget_s', '?')} s; "
            f"{u.get('calls', 0)} model calls, {u.get('input', 0):,} input / {u.get('output', 0):,} output tokens",
            ""]
    if t.get("error"):
        head += ["agent error:", "```", t["error"], "```", ""]
    head += ["## task", _snip(t.get("task"), 1500), "", "## steps"]
    blocks, n_llm = [], 0
    for s in steps:
        if s["kind"] == "llm":
            n_llm += 1
            blocks.append(_llm_block(n_llm, s, t.get("task")))
        elif s["kind"] == "shell":
            blocks.append(_shell_block(s))
        else:
            blocks.append(f"note: {s.get('text')}")
    text = "\n".join(head)
    budget = max_chars - len(text) - 200
    if sum(len(b) + 2 for b in blocks) > budget:
        first = blocks[:3]
        rest, tail = budget - sum(len(b) + 2 for b in first), []
        for b in reversed(blocks[3:]):
            if len(b) + 2 > rest:
                break
            tail.insert(0, b)
            rest -= len(b) + 2
        omitted = len(blocks) - len(first) - len(tail)
        blocks = first + [f"[... {omitted} steps omitted ...]"] + tail
    end = f"\n\n## end\nrun ended: {t.get('end')}; verifier reward {rec.get('reward')}"
    return text + "\n" + "\n\n".join(blocks) + end


def write(records: list[dict[str, Any]], out_dir: Path, *, include: str = "all", max_traces: int | None = None) -> list[Path]:
    """Failed transcripts first, then passed ones (`include="failed"`: failed only), at most `max_traces`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {"fail": 0, "pass": 1} if include == "all" else {"fail": 0}
    scored = sorted((r for r in records if r["outcome"] in order), key=lambda r: (order[r["outcome"]], r["task"]))[:max_traces]
    counts: Counter[str] = Counter()
    rows, paths = [], []
    for r in scored:
        counts[r["task"]] += 1
        p = out_dir / f"{r['task']}__{counts[r['task']]}.md"
        p.write_text(render(r))
        paths.append(p)
        t = r.get("trajectory") or {}
        rows.append(f"| {r['task']} | {r['outcome'].upper()} | {t.get('end', '-')} | {(t.get('usage') or {}).get('calls', 0)} | {t.get('agent_s', 0):.0f} s | {p.name} |")
    index = ["# Trajectories of the current agent on the search tasks", "",
             "| task | result | run ended | model calls | agent time | file |", "|---|---|---|---|---|---|", *rows]
    (out_dir / "INDEX.md").write_text("\n".join(index) + "\n")
    return paths


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
