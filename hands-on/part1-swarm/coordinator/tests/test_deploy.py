from __future__ import annotations

import json

from swarm_coordinator.deploy import stage


def test_stage_holds_everything_the_image_needs(tmp_path):
    out = stage(tmp_path / "space")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    for need in ["Dockerfile", "README.md", "coordinator/pyproject.toml", "coordinator/uv.lock",
                 "coordinator/src/swarm_coordinator/app.py", "coordinator/src/swarm_coordinator/static/board.html",
                 "coordinator/src/swarm_coordinator/data/harnesses.json",
                 "config/grid.json", "config/prices.json", "replay/reference.json"]:
        assert need in names, need
    assert not any("/tests/" in n or ".venv" in n or "__pycache__" in n for n in names)
    readme = (out / "README.md").read_text()
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    docker = (out / "Dockerfile").read_text()
    assert "SWARM_ROOT=/home/user/app" in docker and '"--host", "0.0.0.0"' in docker
    assert json.loads((out / "config/grid.json").read_text())["harnesses"]


def test_stage_never_copies_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_THISmustNOTleak")
    out = stage(tmp_path / "space")
    for p in out.rglob("*"):
        if p.is_file():
            assert b"hf_THISmustNOTleak" not in p.read_bytes()
