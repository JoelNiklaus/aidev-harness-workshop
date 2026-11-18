"""Phase 2 patch gallery: the most interesting real patches across the runs, from the run folders.

    uv run --with markdown python devtools/p2_gallery.py --out "<talk>/evolve"

Each entry: the failure it targets (a short excerpt of a real trajectory the proposer read), the diff (changed
lines only, at most 15), a plain-language explanation to say aloud, and the measured effect (validation before ->
after, the verdict, held-out for the kept ones). The selection and the explanations are written here by hand;
every number comes from the run folders (state.json, records.jsonl) and p2-numbers.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "p2"

ENTRIES = [
    {"run": "B1", "cand": "c001", "also": ["B2 c001", "A2 c001 (15 -> time)"], "excerpt": ("B1", "e000", "c000", "sparql-university", "end"),
     "say": "The seed agent stopped after ten commands, whatever the clock said. Here it quit after 18 seconds of its 180, in the middle of "
            "reading the data. The patch lets the clock, not a step counter, end the run. Two independent runs found the same fix in their "
            "first iteration, and it is the one that holds on tasks the loop never saw."},
    {"run": "A1", "cand": "c006", "also": [], "excerpt": ("A1", "e000", "c000", "financial-document-processor", "end"),
     "say": "With a 240-second budget, this run first tried four other ideas and dropped them. Iteration 6 combined two things: more steps "
            "(15 to 40) and a nudge when 80 seconds are left: stop exploring, write the answer file now. It was kept and held on the 8 "
            "held-out tasks, but on all 25 tasks the loop never saw it was no better than the seed (76% against 78%). Most of its gain "
            "probably comes from the step limit, not from the nudge: a good example of why one patch should carry one idea."},
    {"run": "B1", "cand": "c003", "also": ["B2 c002, B2 c004 (variants)"], "excerpt": ("sweep", "seedb4", "seed", "polyglot-c-py", "empty_error"),
     "say": "The seed threw away stderr, so a failed command showed the model only 'exit code: 127' and nothing else. A two-line patch "
            "passes the error messages through. It sounds like an obvious win, yet validation did not improve (88 to 83 percent, within noise): "
            "GLM-5.3-Flash mostly tried another command. A bug can be real and still not be what limits the score."},
    {"run": "B2", "cand": "c002", "also": ["A1 c002", "A2 c003", "B1 c002"], "excerpt": ("B2", "e001", "c001", "financial-document-processor", "cut"),
     "say": "The seed showed only the first 1,000 characters of every output, and errors and results usually come last. The patch keeps "
            "the head and the tail and says how much it cut. All four runs proposed this. It never beat the noise margin on this model "
            "and these tasks."},
    {"run": "B1", "cand": "c004", "also": ["A1 c004", "A2 c002", "B2 c005"], "excerpt": ("B1", "e001", "c001", "sparql-university", "text_stop"),
     "say": "When the model answers in plain text, the seed stops, even if nothing was tested. The patch sends one more message: re-read the "
            "task, run your result, fix what fails, then finish. Every run tried a version of this; none cleared the margin."},
    {"run": "B2", "cand": "c003", "also": ["B1 c007", "A1 c006 (part)"], "excerpt": ("B2", "e001", "c001", "financial-document-processor", "end"),
     "say": "The model never saw the clock. Here it spent almost the whole budget running OCR on one unreadable scan, then ran out "
            "of time. The patch appends the time left to every tool result and, near the end, says: wrap up and write your best answer now. "
            "Validation 85 to 92 percent, below the margin of one task."},
    {"run": "B2", "cand": "c008", "also": ["B2 c007 (checkpoints)"], "excerpt": ("B2", "e001", "c001", "financial-document-processor", "repeat"),
     "say": "A loop detector: if the last four commands all ran the same program, the harness tells the model to change method or move "
            "on. Classic agent-harness code, and worth reading closely: it takes the last program of a pipeline, so four `cat file | head` "
            "commands trip it, while the OCR loop it was written for alternates `tesseract` with `head` and rarely does. Validation 84 to "
            "92 percent, dropped by the rule."},
    {"run": "A1", "cand": "c001", "also": [], "excerpt": ("A1", "e000", "c000", "dna-insert", "text_stop"),
     "say": "The first idea of run A1: no step limit plus a forced verification turn. It scored 75 to 80 percent, but its trials ran "
            "close to the full 240 seconds and cost about seven times as much as the parent's in the same Job ($0.027 against $0.0035 per trial). The loop dropped it on score alone; cost is a second reason a "
            "harness change has to earn its place."},
]


def records(source: str, ev: str, cand: str, task: str) -> list[dict]:
    path = RUNS / ("sweep-" + ev + "-180.jsonl" if source == "sweep" else f"{source}/records.jsonl")
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["task"] == task and (source == "sweep" or (r.get("eval") == ev and r["candidate"] == cand)):
            out.append(r)
    return out


def one(text: str | None, n: int = 110) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def excerpt(spec: tuple) -> str:
    source, ev, cand, task, mode = spec
    recs = [r for r in records(source, ev, cand, task) if r["outcome"] == "fail"] or records(source, ev, cand, task)
    lines = []
    for r in recs:
        t = r["trajectory"]
        steps = t["steps"]
        shells = [s for s in steps if s["kind"] == "shell"]
        llms = [s for s in steps if s["kind"] == "llm"]
        head = f"[{task}, {r['candidate'] if source != 'sweep' else 'seed'}] "
        if mode == "end" and shells:
            for s in shells[-2:]:
                lines.append(f"$ {one(s['command'])}")
                lines.append(f"  {one(s['output'], 90)}")
            lines.insert(0, head + f"{len(llms)} model calls, {t['agent_s']:.0f} s of {t['budget_s']:.0f} s used, run ended: {t['end']}")
            why = ("time budget used up" if t["agent_s"] >= t["budget_s"] - 15 else "step limit") if llms and llms[-1]["reply"].get("tool_calls") else t["end"]
            lines.append(f"(run ends here: {why}; verifier: fail)")
            break
        if mode == "empty_error":
            hit = next((s for s in shells if s["exit_code"] not in (0, None, -1) and not s["output"]), None)
            if hit:
                lines += [head + "a command fails; the model sees:", f"$ {one(hit['command'])}", f"exit code: {hit['exit_code']}", "(nothing else: stderr went to /dev/null)"]
                break
        if mode == "cut":
            hit = next((s for s in shells if (s["output_chars"] or 0) > 1000), None)
            if hit:
                tail = (hit["output"] or "")[-160:]
                lines += [head + f"the command printed {hit['output_chars']:,} chars; the model saw the first 1,000", f"$ {one(hit['command'])}",
                          f"never shown, the end of the output: …{one(tail, 150)}"]
                break
        if mode == "text_stop" and llms and not llms[-1]["reply"].get("tool_calls"):
            lines += [head + f"the model replies in text after {t['agent_s']:.0f} s of {t['budget_s']:.0f} s; the harness stops, the verifier fails",
                      f"model: {one(llms[-1]['reply'].get('content'), 160)}"]
            break
        if mode == "repeat":
            def program(cmd: str) -> str:  # the patch's own rule: the program of the last command in a chain
                cmd = re.sub(r"^\{ |\n\} 2>/dev/null$", "", (cmd or "").strip())
                last = re.split(r"&&|\|\||[|;]", cmd.strip())[-1].split()
                return last[0] if last else ""

            progs = [one(s["command"], 90) for s in shells]
            for i in range(len(shells) - 3):
                if len({program(s["command"]) for s in shells[i:i + 4]}) == 1 and program(shells[i]["command"]):
                    lines += [head + f"four commands in a row end in `{program(shells[i]['command'])}` (commands {i + 1}-{i + 4} of {len(shells)}; "
                                     f"{t['agent_s']:.0f} s used in total; verifier: fail)"]
                    lines += [f"$ {p}" for p in progs[i:i + 4]]
                    break
            if lines:
                break
    return "\n".join(lines) or "(no excerpt found)"


def short_diff(path: Path, limit: int = 15) -> str:
    changed = [l for l in path.read_text().splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    if len(changed) > limit:
        changed = changed[: limit - 1] + [f"… ({len(changed) - limit + 1} more changed lines in the run folder)"]
    return "\n".join(changed)


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def effect(run: str, cand: str, heldout: dict[str, str]) -> str:
    st = json.loads((RUNS / run / "state.json").read_text())
    h = next(x for x in st["history"] if x["candidate"] == cand)
    first = f" (first measurement {pct(h['val_parent_first'])} -> {pct(h['val_child_first'])}, then re-measured)" if "val_child_first" in h else ""
    out = (f"Validation {pct(h.get('val_parent'))} -> {pct(h.get('val_child'))}{first}; margin {pct(h.get('margin'))}; verdict **{h['verdict']}**; "
           f"iteration {h['iteration']} of run {run}, {h['iter_s'] / 60:.1f} min, ${h['iter_usd']:.2f}.")
    if st["incumbent"] == cand and run in heldout:
        out += f" Held-out (all measurements pooled): {heldout[run]}."
    return out


def heldout_lines(numbers: Path) -> dict[str, str]:
    """Run -> its held-out line from p2-numbers.md (written by p2_report.py)."""
    out = {}
    if numbers.exists():
        for line in numbers.read_text().splitlines():
            m = re.match(r"\| (\w) \| (\w+) final \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", line)
            if m:
                out[m.group(2)] = f"{m.group(3).strip()}, {m.group(4).strip()} over the seed, 95% CI {m.group(6).strip()} (trials resampled), {m.group(5).strip()} (tasks resampled)"
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    ho = heldout_lines(out / "p2-numbers.md")
    md = ["# Patch gallery: what the evolve loop wrote (Phase 2, GLM-5.3-Flash, Terminal-Bench 2.1)", "",
          "Real patches from four loops of 8 iterations (runs A1, A2: Phase 1 seed, 240 s per task; runs B1, B2: the four-weakness "
          "seed, 180 s per task). Proposer: Claude Code (Sonnet). Each patch was measured by the loop on validation tasks the "
          "proposer never sees (2 trials per task for child and parent; kept only above the noise margin and again after a re-measure). "
          "Numbers: `p2-numbers.md` (generated). Figures: `p2-validation.png` (validation pass rate of each run's best agent; "
          "it moves as re-measurements pool in), `p2-heldout.png` (gain of each run's final agent over its seed on every pool task "
          "the loop never saw, 24-25 tasks x 2 trials; whiskers: 95% CI, tasks resampled).", "",
          "**Bottom line.** Config B (the default now) shows a large, visible jump in iteration 1 in both runs (validation 33-42% -> "
          "83-88%), and its gain holds: +14 to +18 points on its 8 held-out tasks and +15 to +17 points on all 24 tasks the loop never "
          "saw (95% CI over tasks [+2, +31]). Config A's kept patches looked as good on its 8 held-out tasks but did not generalize: "
          "on all 25 unseen tasks its finals scored 2-4 points below the seed.", ""]
    numbers = out / "p2-numbers.md"
    if numbers.exists():
        md += ["## Results in one table", "", numbers.read_text().split("\n", 2)[2], ""]
    for i, e in enumerate(ENTRIES, 1):
        d = RUNS / e["run"] / "candidates" / e["cand"]
        title = (d / "PATCH.md").read_text().splitlines()[0].lstrip("# ").strip()
        md += [f"## {i}. {title} ({e['run']} {e['cand']})", ""]
        if e["also"]:
            md += [f"Also proposed by: {', '.join(e['also'])}.", ""]
        md += ["**The failure it targets** (a real trajectory the proposer read):", "", "```text", excerpt(e["excerpt"]), "```", "",
               "**The patch** (changed lines):", "", "```diff", short_diff(d / "diff.patch"), "```", "",
               f"**Say it aloud:** {e['say']}", "", f"**Measured:** {effect(e['run'], e['cand'], ho)}", ""]
    md += ["## How to read the verdicts", "",
           "- The margin is one validation task or 1.5 paired standard errors, whichever is larger (computed from the seed's own "
           "run-to-run flips). Once the step limit is fixed, validation sits at 83-88% and no single patch can add a full task "
           "plus noise; this is why later ideas were dropped even at +8 or +16 points (B2 c007: 84 -> 100%, below its 25-point margin).",
           "- A kept patch is measured twice (child and parent again) and decided on everything pooled, so one lucky Job cannot make it "
           "the new parent.", ""]
    text = "\n".join(md)
    (out / "patch-gallery.md").write_text(text)
    import markdown

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    css = ("body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;line-height:1.45;color:#0F172A}"
           "pre{background:#F1F5F9;padding:.8em;white-space:pre-wrap;word-break:break-word;font-size:.85em}"
           "table{border-collapse:collapse;font-size:.85em}td,th{border:1px solid #CBD5E1;padding:.3em .5em;vertical-align:top}"
           "code{word-break:break-word}img{max-width:100%}")
    figs = "".join(f'<p><img src="{f}" alt="{f}"></p>' for f in ("p2-validation.png", "p2-heldout.png") if (out / f).exists())
    (out / "patch-gallery.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Patch gallery</title><style>{css}</style>{figs}{body}")
    print(out / "patch-gallery.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
