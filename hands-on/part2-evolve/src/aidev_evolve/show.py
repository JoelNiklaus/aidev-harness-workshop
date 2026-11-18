"""Terminal rendering: plain strings (ANSI colour optional), so the loop, replay and tests share them."""

from __future__ import annotations

from typing import Any

RESET, BOLD, DIM, RED, GREEN, YELLOW, CYAN = "\033[0m", "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m"
BARS = " ▁▂▃▄▅▆▇█"
MAX_DIFF_LINES = 60
HYPOTHESIS_WIDTH = 150  # one line on a laptop terminal


def paint(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{RESET}" if color else text


def pct(x: float | None) -> str:
    return "  -  " if x is None else f"{100 * x:4.0f}%"


def trial_line(rec: dict[str, Any], split: str, color: bool = False) -> str:
    mark = {"pass": paint("PASS", GREEN, color), "fail": paint("fail", RED, color), "infra": paint("INFRA", YELLOW, color)}[rec["outcome"]]
    end = (rec.get("trajectory") or {}).get("end", "?")
    return f"    {mark:5} {rec['candidate']:>5} {split:<10} {rec['task']:<30} {rec.get('wall_s') or 0:4.0f} s  {end}\n"


def spend_line(spent: float, proposer_usd: float | None, color: bool = False) -> str:
    sub = f"; proposer on your own subscription (API-price equivalent): ${proposer_usd:.2f}" if proposer_usd else ""
    return paint(f"  HF credits spent so far: ${spent:.3f}{sub}\n", DIM, color)


def chart(points: list[float]) -> str:
    return "".join(BARS[min(len(BARS) - 1, round(p * (len(BARS) - 1)))] for p in points)


def seed_summary(search: float, val: float, flip: float, margin: float, n_val: int, color: bool = False) -> str:
    return (f"\n{paint('Seed agent', BOLD, color)}: search {pct(search)}, validation {pct(val)} (two trials per validation task)\n"
            f"  noise: {100 * flip:.0f}% of validation tasks flip between two runs of the same agent -> keep a patch only if it gains "
            f"at least {100 * margin:.0f} points ({margin * n_val:.1f} of {n_val} tasks)\n")


def diff_block(diff: str, color: bool = False) -> str:
    lines = diff.splitlines()
    shown = lines[:MAX_DIFF_LINES]
    out = []
    for line in shown:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(paint(line, GREEN, color))
        elif line.startswith("-") and not line.startswith("---"):
            out.append(paint(line, RED, color))
        elif line.startswith("@@"):
            out.append(paint(line, CYAN, color))
        else:
            out.append(line)
    if len(lines) > MAX_DIFF_LINES:
        out.append(paint(f"... {len(lines) - MAX_DIFF_LINES} more diff lines in the run folder", DIM, color))
    return "\n".join("    " + line for line in out) + "\n"


def one_line(text: str, width: int = HYPOTHESIS_WIDTH) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def proposal_block(h: dict[str, Any], color: bool = False) -> str:
    """What the proposer changed, shown as soon as it is done, so the attendee reads it while the Job runs."""
    out = f"  proposer ({h.get('proposer_s', 0):.0f} s): {paint(h.get('title') or '', BOLD, color)}\n"
    if h.get("hypothesis"):
        out += f"  {paint(one_line(h['hypothesis']), CYAN, color)}\n"
    if h.get("diff"):
        out += diff_block(h["diff"], color)
    return out


def iteration_card(h: dict[str, Any], curve: list[float], color: bool = False) -> str:
    """The result of one iteration (the diff was shown by proposal_block before the evaluation)."""
    head = f"\n{paint(f'Iteration {h['iteration']}', BOLD, color)}  {h['candidate']} from {h['parent']}: {paint(h.get('title') or '', BOLD, color)}\n"
    body = ""
    verdict = h["verdict"]
    if "val_child" in h:  # evaluated: kept, dropped, or dropped after the re-measure
        delta = h["val_child"] - h["val_parent"]
        reason = verdict.partition(":")[2].strip()  # "dropped: did not hold on re-measure" -> the reason
        word = paint("KEPT", GREEN, color) if verdict == "kept" else paint("DROPPED" + (f" ({reason})" if reason else ""), RED, color)
        if "val_child_first" in h:
            body += f"  first measurement {pct(h['val_parent_first'])} -> {pct(h['val_child_first'])}, then re-measured (pooled):\n"
        body += (f"  validation {pct(h['val_parent'])} -> {pct(h['val_child'])} ({100 * delta:+.0f} points, needs {100 * h['margin']:+.0f})  {word}\n"
                 f"  fixed: {', '.join(h['fixed']) or '-'}    broke: {', '.join(h['broke']) or '-'}    search: {pct(h.get('search_child'))}\n")
    else:
        body += f"  {paint(verdict, YELLOW, color)}" + (f": {h['detail'][:300]}" if h.get("detail") else "") + "\n"
    body += f"  best agent so far, validation: {chart(curve)} {pct(curve[-1] if curve else None)}\n"
    return head + body


def heldout_summary(res: dict[str, Any], color: bool = False) -> str:
    return (f"\n{paint('Held-out check', BOLD, color)} ({res['n_tasks']} tasks never seen during the search, {res['trials']} trial(s) each):\n"
            f"  seed {pct(res['seed'])} -> best ({res['best_candidate']}) {pct(res['best'])}  ({100 * (res['best'] - res['seed']):+.0f} points)\n"
            + (f"  paired gain {100 * res['diff']:+.0f} points, 95% CI [{100 * res['ci_low']:+.0f}, {100 * res['ci_high']:+.0f}] (tasks resampled)\n"
               if "diff" in res else ""))
