"""The evolve loop: seed -> (propose -> check -> evaluate -> keep or drop) x N -> held-out check.

Adapters are injected so the loop runs the same with fakes (tests), the real HF Job evaluator, a recorded
run (replay) or a different proposer:

- `evaluate(spec, on_record) -> launch.EvalResult`: one HF Job runs the spec's trials.
- `propose(work_dir, prompt) -> proposer.ProposerResult`: the coding agent edits work_dir/agent.py.
- `board.submit(event)`: best effort, the room's /evolve view (PROTOCOL.md).
- `write(text)`: the terminal.

Everything lands in the run folder (layout in DESIGN.md): config.json, state.json, records.jsonl (every
trial with its trajectory), candidates/cNNN/{agent.py, PATCH.md, diff.patch}, transcript.jsonl (what the
terminal showed, with times: the replay source), work/ (the proposer's folder for the current iteration).
"""

from __future__ import annotations

import difflib
import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from importlib import resources
from pathlib import Path
from typing import Any

from aidev_evolve import __version__, audit, digest, proposer, runtime, show, stats
from aidev_evolve import strategy as strategies
from aidev_evolve.launch import EvalJobError
from aidev_evolve.runtime import ShellResult

SEED_SOURCE = resources.files("aidev_evolve.seed").joinpath("agent.py").read_text()
# The keep rule, evidence and proposer settings come from the optimization strategy (strategy.py, presets in
# aidev_evolve/strategies/); the held-out trials and the session length are fixed for everyone.
HELDOUT_TRIALS = strategies.HELDOUT_TRIALS


def default_splits() -> dict[str, Any]:
    return json.loads(resources.files("aidev_evolve.data").joinpath("splits.json").read_text())


def preflight(source: str) -> str | None:
    """Run the candidate once against a scripted model and shell: catches crashes before credits are spent."""
    replies = [{"role": "assistant", "content": None, "tool_calls": [{"id": "pf1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "echo hello"}'}}]}]

    def llm(messages, tools, timeout, **params):
        reply = replies.pop(0) if replies else {"role": "assistant", "content": "Done: the task is complete."}
        return reply, {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0}

    traj = runtime.run_agent(source, "Preflight check: print hello.", shell=lambda c, t: ShellResult(0, "hello\n"), llm=llm, budget_s=30)
    return traj["error"] if traj["end"] == "error" else None


class NullBoard:
    def submit(self, event: dict[str, Any]) -> None:
        return None


class Evolver:
    def __init__(self, run_dir: Path, *, splits: dict[str, Any] | None = None, evaluate: Callable[..., Any], propose: Callable[..., Any],
                 board: Any = None, write: Callable[[str], None] = print, budget_s: float | None = None, seed_source: str = SEED_SOURCE,
                 color: bool = False, run_id: str | None = None, proposer_name: str = "claude", clock: Callable[[], float] = time.monotonic,
                 resume: bool = False, strategy: strategies.Strategy | None = None, session_s: float | None = None,
                 parallel_proposals: bool = False):
        self.run_dir = Path(run_dir)
        self.splits = splits or default_splits()
        self.evaluate, self.propose, self.board = evaluate, propose, board or NullBoard()
        self._write, self.color, self.clock = write, color, clock
        self.budget_s = budget_s or self.splits.get("budget_s")  # the budget the splits were selected for (a resumed run: its config)
        self.session_s, self.parallel_proposals = session_s, parallel_proposals
        self._use(strategy or strategies.load("baseline"))
        self.run_id = run_id or self.run_dir.name
        self.t0 = clock()
        self.records: list[dict[str, Any]] = []
        self.task_names = [t for k in ("search", "validation", "heldout") for t in self.splits.get(k, [])]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume and (self.run_dir / "state.json").exists():
            self._load()
            return
        (self.run_dir / "config.json").write_text(json.dumps({"run_id": self.run_id, "version": __version__, "splits": self.splits, "budget_s": self.budget_s,
                                                              "model": "zai-org/GLM-5.3-Flash", "effort": "low", "proposer": proposer_name,
                                                              "session_s": session_s, "strategy": {**asdict(self.strategy), **self.strategy.event()}}, indent=2))
        self.state: dict[str, Any] = {"run_id": self.run_id, "incumbent": "c000", "iteration": 0, "margin": None, "flip_rate": None,
                                      "history": [], "spend_usd": 0.0, "proposer_usd": 0.0, "heldout": None, "curve": [], "search_eval": {}}
        self._save_candidate("c000", seed_source, None, "", "")

    def _use(self, s: strategies.Strategy) -> None:
        self.strategy = s
        self.val_trials, self.z, self.confirm = s.val_trials, s.z, s.confirm

    def _load(self) -> None:
        """Continue a run from its folder (after a crash or Ctrl-C): state, records and the transcript clock."""
        self.state = json.loads((self.run_dir / "state.json").read_text())
        config = json.loads((self.run_dir / "config.json").read_text())
        self.splits, self.budget_s, self.session_s = config["splits"], config["budget_s"], config.get("session_s")
        self._use(strategies.Strategy(**{f.name: config["strategy"][f.name] for f in fields(strategies.Strategy)}))
        rec_path = self.run_dir / "records.jsonl"
        self.records = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()] if rec_path.exists() else []
        tr = self.run_dir / "transcript.jsonl"
        last = json.loads(tr.read_text().splitlines()[-1])["t"] if tr.exists() and tr.read_text().strip() else 0.0
        self.t0 = self.clock() - last
        self.run_id = self.state["run_id"]

    # -- plumbing -----------------------------------------------------------------------------------
    def write(self, text: str) -> None:
        self._write(text)
        with open(self.run_dir / "transcript.jsonl", "a") as f:
            f.write(json.dumps({"t": round(self.clock() - self.t0, 2), "text": text}) + "\n")

    def _save(self) -> None:
        (self.run_dir / "state.json").write_text(json.dumps(self.state, indent=2))

    def _cand_dir(self, cand: str) -> Path:
        return self.run_dir / "candidates" / cand

    def source(self, cand: str) -> str:
        return (self._cand_dir(cand) / "agent.py").read_text()

    def _save_candidate(self, cand: str, src: str, parent: str | None, patch: str, diff: str) -> None:
        d = self._cand_dir(cand)
        d.mkdir(parents=True, exist_ok=True)
        (d / "agent.py").write_text(src)
        if parent:
            (d / "PATCH.md").write_text(patch)
            (d / "diff.patch").write_text(diff)

    def _split(self, task: str) -> str:
        for k in ("search", "validation", "heldout"):
            if task in self.splits.get(k, []):
                return k
        return "other"

    def _submit(self, event: dict[str, Any]) -> None:
        try:
            waiting = self.board.submit({"run_id": self.run_id, **event, "strategy": self.strategy.event()})
        except Exception as e:  # noqa: BLE001 - the workshop board never stops the loop
            self.write(show.paint(f"  (workshop board unreachable: {type(e).__name__})\n", show.DIM, self.color))
            return
        if waiting:  # a QueuedBoard keeps undelivered events in the run folder and retries with the next one
            self.write(show.paint(f"  (workshop board offline: {waiting} updates queued, sent when it is back; your loop goes on)\n", show.DIM, self.color))

    def _eval(self, eval_id: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cands = {r["candidate"] for r in runs}
        spec = {"candidates": {c: self.source(c) for c in cands}, "runs": runs, "budget_s": self.budget_s}
        n = sum(len(r["tasks"]) * r["trials"] for r in runs)
        self.write(f"  evaluating {n} trials in one HF Job ({', '.join(sorted(cands))})...\n")

        def on_record(rec: dict[str, Any]) -> None:
            self.write(show.trial_line(rec, self._split(rec["task"]), self.color))

        res = self.evaluate(spec, on_record)
        cost = sum(r.get("cost_usd") or 0.0 for r in res.records) + res.job_usd
        self.state["spend_usd"] += cost
        with open(self.run_dir / "records.jsonl", "a") as f:
            for r in res.records:
                r = {**r, "eval": eval_id, "split": self._split(r["task"])}
                self.records.append(r)
                f.write(json.dumps(r) + "\n")
        self.write(f"  Job {res.job_id}: {res.seconds:.0f} s, ${cost:.3f}\n")
        if getattr(res, "missing", 0):
            self.write(show.paint(f"  warning: {res.missing} planned trials sent no result (see the Job log); they do not count\n", show.YELLOW, self.color))
        return [r for r in self.records if r["eval"] == eval_id]

    def val_score(self, cand: str) -> float:
        return stats.score(stats.per_task(self.records, cand, self.splits["validation"]))

    def search_score(self, cand: str) -> float:
        return stats.score(stats.per_task(self.records, cand, self.splits["search"]))

    # -- phases -------------------------------------------------------------------------------------
    def calibrate_seed(self) -> None:
        self.write(show.paint(f"\nEvolve run {self.run_id}: seed agent on {len(self.splits['search'])} search + {len(self.splits['validation'])} validation tasks\n", show.BOLD, self.color))
        self._eval("e000", [{"candidate": "c000", "tasks": self.splits["search"], "trials": 1},
                            {"candidate": "c000", "tasks": self.splits["validation"], "trials": 2}])
        flip = stats.flip_rate(stats.per_task(self.records, "c000", self.splits["validation"]))
        n = len(self.splits["validation"])
        self.state.update(flip_rate=flip, margin=stats.margin(flip, n, self.z, self.val_trials))
        self.state["search_eval"]["c000"] = "e000"
        self.state["curve"] = [self.val_score("c000")]
        self.write(show.seed_summary(self.search_score("c000"), self.val_score("c000"), flip, self.state["margin"], n, self.color))
        self.write(show.spend_line(self.state["spend_usd"], None, self.color))
        self._submit({"kind": "seed", "val_score": self.val_score("c000"), "search_score": self.search_score("c000"), "spend_usd": self.state["spend_usd"]})
        self._save()

    def _history_md(self) -> str:
        mode, diffs = self.strategy.history, self.strategy.diffs
        lines = ["# Earlier iterations", ""]
        if mode == "none":
            return "\n".join(lines + ["(this strategy does not show earlier attempts)"]) + "\n"
        if not self.state["history"]:
            lines.append("(none yet: this is the first proposal)")
        for h in self.state["history"]:
            score = f"validation {100 * h['val_parent']:.0f}% -> {100 * h['val_child']:.0f}%" if h.get("val_child") is not None else ""
            lines.append(f"## iteration {h['iteration']}: {h.get('title')} — {h['verdict']} {score}")
            for sib in h.get("siblings", []):
                val = f" (validation {100 * sib['val']:.0f}%)" if sib.get("val") is not None else ""
                lines.append(f"also proposed: {sib['title']} — {sib['verdict']}{val}")
            if mode == "full":
                lines += [(h.get("patch") or "").strip(), f"fixed: {', '.join(h.get('fixed', [])) or '-'}; broke: {', '.join(h.get('broke', [])) or '-'}"]
            if diffs and h.get("diff"):
                lines += ["```diff", h["diff"].strip(), "```"]
            lines.append("")
        return "\n".join(lines) + "\n"

    def _prepare_work(self, iteration: int, name: str = "work", hint: str = "") -> Path:
        work = self.run_dir / name
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        inc = self.state["incumbent"]
        (work / "agent.py").write_text(self.source(inc))
        ev = self.state["search_eval"][inc]
        digest.write([r for r in self.records if r["eval"] == ev and r["candidate"] == inc and r["split"] == "search"], work / "traces",
                     include=self.strategy.traces, max_traces=self.strategy.max_traces)
        (work / "HISTORY.md").write_text(self._history_md())
        (work / "CONTRACT.md").write_text(proposer.CONTRACT)
        prompt = proposer.build_prompt(iteration=iteration, budget_s=self.budget_s, approach=self.strategy.prompt, hint=hint)
        (work / "PROMPT.md").write_text(prompt)
        return work

    def _child(self, work: Path, parent: str, cand: str, pres: Any) -> dict[str, Any]:
        """One proposal read back and checked: audit and preflight before any credits are spent."""
        self.state["proposer_usd"] += pres.cost_usd or 0.0
        new_src = (work / "agent.py").read_text()
        title, patch = proposer.read_patch_notes(work)
        diff = "".join(difflib.unified_diff(self.source(parent).splitlines(True), new_src.splitlines(True), f"agent.py ({parent})", f"agent.py ({cand})"))
        c: dict[str, Any] = {"candidate": cand, "title": title, "patch": patch, "diff": diff, "source": new_src, "proposer_s": round(pres.seconds, 1),
                             "hypothesis": next((l for l in patch.splitlines() if l.startswith("Hypothesis")), "")}
        if pres.ok:
            self.write(show.proposal_block({"iteration": self.state["iteration"], "candidate": cand, "parent": parent, **c}, self.color))
        if not pres.ok:
            c.update(verdict="skipped: proposer failed", detail=pres.error)
        elif new_src == self.source(parent):
            c.update(verdict="skipped: agent.py unchanged")
        elif problems := audit.problems(new_src, self.task_names):
            c.update(verdict="rejected: audit", detail="; ".join(problems))
        elif crash := preflight(new_src):
            c.update(verdict="rejected: crashed in preflight", detail=crash)
        return c

    def _proposals(self, i: int, parent: str) -> list[dict[str, Any]]:
        n = self.strategy.candidates
        names = [f"c{i:03d}"] if n == 1 else [f"c{i:03d}{x}" for x in "ab"[:n]]
        hints = ["", self.strategy.second_hint][:n]
        works = [self._prepare_work(i, "work" if n == 1 else f"work-{name[-1]}", hint) for name, hint in zip(names, hints)]

        def one(w: Path) -> Any:
            return self.propose(w, (w / "PROMPT.md").read_text())

        if self.parallel_proposals and n > 1:
            with ThreadPoolExecutor(n) as pool:
                results = list(pool.map(one, works))
        else:
            results = [one(w) for w in works]
        return [self._child(w, parent, name, r) for w, name, r in zip(works, names, results)]

    def iterate(self) -> dict[str, Any]:
        self.state["iteration"] += 1
        i, parent = self.state["iteration"], self.state["incumbent"]
        self._iter_start, self._iter_spend = self.clock(), self.state["spend_usd"]
        self.write(show.paint(f"\n--- iteration {i}: proposing a change to {parent} ---\n", show.BOLD, self.color))
        children = self._proposals(i, parent)
        valid = [c for c in children if "verdict" not in c]

        def sibling(c: dict[str, Any], val: float | None = None, verdict: str | None = None) -> dict[str, Any]:
            return {"candidate": c["candidate"], "title": c["title"], "val": val, "verdict": verdict or c["verdict"]}

        def entry(c: dict[str, Any], **extra: Any) -> dict[str, Any]:
            h = {"iteration": i, "parent": parent, **{k: v for k, v in c.items() if k != "source"}, **extra}
            h["siblings"] = [sibling(o) for o in children if o is not c and "verdict" in o] + h.get("siblings", [])
            return h

        if not valid:
            h = entry(children[0])
            self._finish(h)
            return h
        for c in valid:
            self._save_candidate(c["candidate"], c["source"], parent, c["patch"], c["diff"])
        ev = f"e{i:03d}"
        vt, val, search = self.val_trials, self.splits["validation"], self.splits["search"]
        runs: list[dict[str, Any]] = []
        for c in valid:
            runs += ([{"candidate": c["candidate"], "tasks": search + val, "trials": 1}] if vt == 1 else
                     [{"candidate": c["candidate"], "tasks": search, "trials": 1}, {"candidate": c["candidate"], "tasks": val, "trials": vt}])
        try:
            self._eval(ev, runs + [{"candidate": parent, "tasks": val, "trials": vt}])
        except EvalJobError as e:
            h = entry(valid[0], verdict="skipped: evaluation failed", detail=str(e)[:500])
            self._finish(h)
            return h
        decisions = {c["candidate"]: stats.decide(self.records, c["candidate"], parent, self.state["margin"], val) for c in valid}
        best = max(valid, key=lambda c: decisions[c["candidate"]].child_score)  # first wins a tie
        cand, d = best["candidate"], decisions[best["candidate"]]
        losers = [sibling(c, decisions[c["candidate"]].child_score, "dropped: lost the screen") for c in valid if c is not best]
        verdict = "kept" if d.keep else "dropped"
        h = entry(best, siblings=losers)
        if d.keep and (self.confirm or len(valid) > 1):
            # one lucky measurement (or the luckier of two) must not become the bar: measure child and parent again, decide on everything pooled
            h.update(val_child_first=d.child_score, val_parent_first=d.parent_score)
            self.write(f"  validation {100 * d.parent_score:.0f}% -> {100 * d.child_score:.0f}%: re-measuring before it counts...\n")
            try:
                self._eval(f"e{i:03d}r", [{"candidate": cand, "tasks": val, "trials": vt}, {"candidate": parent, "tasks": val, "trials": vt}])
            except EvalJobError as e:
                h.update(verdict="skipped: evaluation failed", detail=str(e)[:500])
                self._finish(h)
                return h
            d = stats.decide(self.records, cand, parent, self.state["margin"], val)
            verdict = "kept" if d.keep else "dropped: did not hold on re-measure"
        h.update(verdict=verdict, val_child=d.child_score, val_parent=d.parent_score, margin=d.margin,
                 fixed=d.fixed, broke=d.broke, search_child=self.search_score(cand))
        if d.keep:
            self.state["incumbent"] = cand
            self.state["search_eval"][cand] = ev
        self._finish(h)
        return h

    def _finish(self, h: dict[str, Any]) -> None:
        self.state["curve"].append(self.val_score(self.state["incumbent"]))
        h["best_val"] = self.state["curve"][-1]
        h["spend_usd"] = round(self.state["spend_usd"], 6)
        h["iter_usd"] = round(self.state["spend_usd"] - self._iter_spend, 6)
        h["iter_s"] = round(self.clock() - self._iter_start, 1)
        self.state["history"].append(h)
        self.write(show.iteration_card(h, self.state["curve"], self.color))
        for sib in h.get("siblings", []):
            val = f", validation {100 * sib['val']:.0f}%" if sib.get("val") is not None else ""
            self.write(show.paint(f"  also proposed {sib['candidate']}: {sib['title']} — {sib['verdict']}{val}\n", show.DIM, self.color))
        self.write(show.spend_line(self.state["spend_usd"], self.state["proposer_usd"], self.color))
        self._submit({"kind": "iteration", "iteration": h["iteration"], "candidate": h["candidate"], "parent": h["parent"], "title": h.get("title"),
                      "accepted": h["verdict"] == "kept", "verdict": h["verdict"], "val_child": h.get("val_child"), "val_parent": h.get("val_parent"),
                      "best_val": h["best_val"], "fixed": h.get("fixed", []), "broke": h.get("broke", []), "diff": (h.get("diff") or "")[:8000],
                      "hypothesis": h.get("hypothesis") or None, "siblings": h.get("siblings", []), "spend_usd": h["spend_usd"]})
        self._save()

    def heldout(self, trials: int = 1) -> dict[str, Any]:
        best = self.state["incumbent"]
        tasks = self.splits["heldout"]
        self.write(show.paint(f"\n--- held-out check: seed vs {best} on {len(tasks)} unseen tasks ---\n", show.BOLD, self.color))
        runs = [{"candidate": "c000", "tasks": tasks, "trials": trials}]
        if best != "c000":
            runs.append({"candidate": best, "tasks": tasks, "trials": trials})
        recs = self._eval("heldout", runs)
        pd = stats.paired_diff(recs, best, "c000", tasks)
        seed_pt, best_pt = stats.per_task(recs, "c000", tasks), stats.per_task(recs, best, tasks)
        counts = {t: [sum(seed_pt[t]), len(seed_pt[t]), sum(best_pt[t]), len(best_pt[t])] for t in tasks if t in seed_pt and t in best_pt}
        kept = [h["title"] for h in self.state["history"] if h["verdict"] == "kept"]
        res = {"seed": stats.score(seed_pt), "best": stats.score(best_pt), "best_candidate": best, "n_tasks": len(tasks), "trials": trials,
               "diff": pd.diff, "ci_low": pd.ci_low, "ci_high": pd.ci_high, "per_task": counts, "kept": kept}
        self.state["heldout"] = res
        self.write(show.heldout_summary(res, self.color))
        self.write(show.spend_line(self.state["spend_usd"], self.state["proposer_usd"], self.color))
        self._submit({"kind": "heldout", **res, "spend_usd": self.state["spend_usd"]})
        self._save()
        return res

    def run(self, iterations: int | None = None, heldout_trials: int = 0) -> dict[str, Any]:
        """Run until `iterations` iterations exist in total (default: the strategy's; a resumed run continues where it
        stopped), or until the session time box leaves no room for another iteration and the held-out check."""
        iterations = self.strategy.iterations if iterations is None else iterations
        if self.state["margin"] is None:
            self.calibrate_seed()
        else:
            self.write(show.paint(f"\nResuming {self.run_id} after iteration {self.state['iteration']} (best: {self.state['incumbent']})\n", show.BOLD, self.color))
        while self.state["iteration"] < iterations:
            if self.session_s and not strategies.next_fits(self.strategy, elapsed_s=self.clock() - self.t0, session_s=self.session_s,
                                                           iteration_s=[h["iter_s"] for h in self.state["history"]  # a re-measure is not the norm
                                                                        if "iter_s" in h and "val_child_first" not in h]):
                self.write(show.paint(f"\nsession time box ({self.session_s / 60:.0f} min): no time for another iteration and the held-out check; "
                                      f"stopping after {self.state['iteration']} iterations\n", show.DIM, self.color))
                break
            self.iterate()
        if heldout_trials:
            self.heldout(heldout_trials)
        return self.state
