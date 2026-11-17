"""devtools/dryrun_report.py: parse runner Job logs and summarize a dry run (pure parts, real log lines)."""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location("dryrun_report", Path(__file__).parents[1] / "devtools" / "dryrun_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

LOG = """\
Installed 94 packages in 330ms
[aidev-swarm 11:06:46] claimed pi on build-pmars (trial 0), model zai-org/GLM-5.3-Flash: starting an HF Sandbox
  1/1 Mean: 0.000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:01:13 0:00:00
[aidev-swarm 11:07:58] PASS pi build-pmars: 9 model calls, 23,704 tokens, $0.0035, 63 s. Submitted. Board: https://x--8765.hf.jobs
[aidev-swarm 11:07:58] claimed opencode on hf-model-inference (trial 0), model zai-org/GLM-5.3-Flash: starting an HF Sandbox
[aidev-swarm 11:09:01] trial crashed (RuntimeError: agent setup failed)
[aidev-swarm 11:09:02] skipped after its setup failure: cell released
[aidev-swarm 11:09:02] claimed openclaw on pypi-server (trial 0), model zai-org/GLM-5.3-Flash: starting an HF Sandbox
[aidev-swarm 11:12:00] infrastructure failure before any model call (EnvironmentStartTimeoutError: pull)
[aidev-swarm 11:12:01] nothing left to claim: the grid is full. Board: https://x--8765.hf.jobs
"""

DAY = datetime(2026, 11, 17, tzinfo=timezone.utc)


def test_parse_log_finds_claims_verdicts_and_failures_with_utc_times():
    ev = rep.parse_log(LOG, DAY)
    assert [(e["kind"], e.get("harness"), e.get("task")) for e in ev] == [
        ("claimed", "pi", "build-pmars"), ("PASS", "pi", "build-pmars"),
        ("claimed", "opencode", "hf-model-inference"), ("crash", None, None),
        ("claimed", "openclaw", "pypi-server"), ("infra", None, None), ("grid_full", None, None)]
    assert ev[0]["at"] == datetime(2026, 11, 17, 11, 6, 46, tzinfo=timezone.utc)
    assert ev[3]["detail"].startswith("RuntimeError: agent setup failed")


def test_failure_causes_are_counted_by_their_exception_type():
    ev = rep.parse_log(LOG, DAY)
    assert rep.failure_causes([ev]) == {"crash: RuntimeError": 1, "infra: EnvironmentStartTimeoutError": 1}


def test_percentiles_of_a_small_sample():
    assert rep.pct([1, 2, 3, 4, 5], 50) == 3 and rep.pct([10], 90) == 10 and rep.pct([], 50) is None


def test_a_cancelled_sandbox_without_finished_at_is_billed_until_the_trial_ended():
    from types import SimpleNamespace

    start = datetime(2026, 11, 17, 11, 7, 3, tzinfo=timezone.utc)
    job = SimpleNamespace(flavor="cpu-basic", started_at=start, finished_at=None, status=SimpleNamespace(expose_urls=["https://x--49983.hf.jobs"]))
    end = start.replace(hour=12)  # one hour later
    assert rep._cost(job, end) == 0.02  # cpu-basic $0.01/h + exposed port $0.01/h
