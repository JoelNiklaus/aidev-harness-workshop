"""Static checks on a proposed agent.py before any credits are spent on it (generality and no escape hatches)."""

from importlib import resources

from aidev_evolve import audit

SEED = resources.files("aidev_evolve.seed").joinpath("agent.py").read_text()


def test_the_seed_passes():
    assert audit.problems(SEED) == []


def test_syntax_errors_and_a_missing_run_are_reported():
    assert "syntax" in audit.problems("def run(task, env)\n    pass\n")[0]
    assert any("run(task, env)" in p for p in audit.problems("x = 1\n"))


def test_only_harmless_standard_modules_may_be_imported():
    probs = audit.problems("import os\nimport subprocess\nfrom urllib import request\nimport json, re\n" + SEED)
    assert len(probs) == 3 and all("import" in p for p in probs)


def test_no_back_doors_to_the_job():
    src = SEED + "\ndef x(env):\n    open('/proc/self/environ').read()\n    eval('1')\n    __import__('os')\n    env._llm\n"
    probs = audit.problems(src)
    for word in ("open", "eval", "__import__", "_llm"):
        assert any(word in p for p in probs), word


def test_private_attributes_by_name_are_refused_too():
    probs = audit.problems(SEED + "\ndef x(env):\n    getattr(env, '_shell')\n    getattr(env, 'time_left')\n")
    assert len(probs) == 1 and "_shell" in probs[0]


def test_benchmark_task_names_are_refused():
    probs = audit.problems(SEED + "\nHINT = 'if the task is sparql-university, use rdflib'\n", task_names=["sparql-university", "fix-git"])
    assert probs == ["names a benchmark task (sparql-university): harness changes must be general"]
