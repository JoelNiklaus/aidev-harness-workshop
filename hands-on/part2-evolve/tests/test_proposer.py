"""The proposer: the attendee's own coding agent edits agent.py (Claude Code, Codex, or by hand)."""

import json
import subprocess

import pytest

from aidev_evolve import proposer


class FakeRun:
    def __init__(self, stdout="", returncode=0, timeout=False):
        self.calls = []
        self.stdout, self.returncode, self.timeout = stdout, returncode, timeout

    def __call__(self, argv, cwd, env, timeout):
        self.calls.append({"argv": argv, "cwd": cwd, "env": env, "timeout": timeout})
        if self.timeout:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "")


def test_claude_runs_headless_with_edit_tools_only_and_without_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    run = FakeRun(stdout=json.dumps({"result": "done", "is_error": False, "total_cost_usd": 0.31}))
    res = proposer.propose("claude", tmp_path, "PROMPT", run=run, which=lambda n: f"/usr/bin/{n}", model="sonnet")
    c = run.calls[0]
    assert c["argv"][:3] == ["/usr/bin/claude", "-p", "PROMPT"]
    assert c["argv"][c["argv"].index("--model") + 1] == "sonnet"
    assert c["argv"][c["argv"].index("--allowedTools") + 1] == "Read,Edit,Write,Glob,Grep"
    assert "--setting-sources" in c["argv"]
    assert "ANTHROPIC_API_KEY" not in c["env"] and c["cwd"] == tmp_path
    assert res.ok and res.cost_usd == pytest.approx(0.31)


def test_claude_error_result_is_not_ok(tmp_path):
    run = FakeRun(stdout=json.dumps({"result": "You've hit your limit", "is_error": True}), returncode=1)
    res = proposer.propose("claude", tmp_path, "P", run=run, which=lambda n: n)
    assert not res.ok and "limit" in res.error


def test_codex_runs_exec_with_workspace_write(tmp_path):
    run = FakeRun(stdout="ok")
    res = proposer.propose("codex", tmp_path, "PROMPT", run=run, which=lambda n: f"/bin/{n}")
    argv = run.calls[0]["argv"]
    assert argv[:2] == ["/bin/codex", "exec"] and "workspace-write" in argv and "--skip-git-repo-check" in argv
    assert argv[-1] == "PROMPT" and res.ok


def test_a_proposer_that_runs_too_long_is_stopped(tmp_path):
    res = proposer.propose("claude", tmp_path, "P", run=FakeRun(timeout=True), which=lambda n: n, timeout_s=5)
    assert not res.ok and "5 s" in res.error


def test_missing_cli_says_how_to_continue(tmp_path):
    res = proposer.propose("codex", tmp_path, "P", run=FakeRun(), which=lambda n: None)
    assert not res.ok and "--proposer manual" in res.error


def test_manual_waits_for_the_attendee(tmp_path):
    asked = []
    res = proposer.propose("manual", tmp_path, "P", ask=lambda msg: asked.append(msg) or "")
    assert res.ok and "PROMPT.md" in asked[0]


def test_patch_notes_title_and_text(tmp_path):
    (tmp_path / "PATCH.md").write_text("# Keep the tail of long outputs\n\nHypothesis: errors are at the end.\n")
    title, text = proposer.read_patch_notes(tmp_path)
    assert title == "Keep the tail of long outputs" and "Hypothesis" in text
    (tmp_path / "PATCH.md").unlink()
    assert proposer.read_patch_notes(tmp_path) == ("(no PATCH.md)", "")


def test_prompt_names_the_files_and_the_rules():
    p = proposer.build_prompt(iteration=3, budget_s=150)
    for word in ("agent.py", "traces/INDEX.md", "HISTORY.md", "PATCH.md", "one", "task names", "150"):
        assert word in p


def test_another_eval_passes_its_own_header_and_keeps_the_fixed_rules():
    """APPLY-TO-YOUR-AGENT.md: the proposer is told about your setting, not about Terminal-Bench."""
    from aidev_evolve import proposer

    p = proposer.build_prompt(iteration=2, budget_s=60, approach="Read agent.py first.", header="My support bot, {budget_s:.0f} s per ticket. Iteration {iteration}.\n")
    assert p.startswith("My support bot, 60 s per ticket. Iteration 2.") and "Terminal-Bench" not in p
    assert "Read agent.py first." in p and "PATCH.md" in p  # the strategy's approach and the fixed rules follow
    assert "Terminal-Bench" in proposer.build_prompt(iteration=1, budget_s=180)  # the workshop's default is unchanged
