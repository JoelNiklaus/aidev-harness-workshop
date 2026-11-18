from __future__ import annotations

import json

import re
import shutil

import pytest

from swarm_coordinator import deploy, paths
from swarm_coordinator.deploy import stage


def test_stage_holds_everything_the_image_needs(tmp_path):
    out = stage(tmp_path / "space")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    for need in ["Dockerfile", "README.md", "coordinator/pyproject.toml", "coordinator/uv.lock",
                 "coordinator/src/swarm_coordinator/app.py", "coordinator/src/swarm_coordinator/static/board.html",
                 "coordinator/src/swarm_coordinator/data/harnesses.json",
                 "config/grid.json", "config/prices.json", "replay/reference.json",
                 "coordinator/src/swarm_coordinator/static/evolve.html", "replay/evolve-recorded.json"]:
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


def test_stage_copies_only_the_app_never_local_files_next_to_it(tmp_path, monkeypatch):
    """A gitignored token file (space-token.txt) or any other local file in coordinator/ never reaches the Space."""
    fake = tmp_path / "coord"
    shutil.copytree(deploy.COORD, fake, ignore=shutil.ignore_patterns(".venv", ".pytest_cache", "__pycache__", "space-token.txt"))
    (fake / "space-token.txt").write_text("SPACE_HF_TOKEN=hf_PLANTED" + "0" * 30 + "\n")  # split: the export's secret scan reads this file
    (fake / "notes-local.md").write_text("private notes")
    monkeypatch.setattr(deploy, "COORD", fake)
    out = stage(tmp_path / "space")
    names = {p.relative_to(out / "coordinator").as_posix() for p in (out / "coordinator").rglob("*") if p.is_file()}
    assert names and all(n in ("pyproject.toml", "uv.lock") or n.startswith("src/") for n in names), sorted(names)
    for p in out.rglob("*"):
        if p.is_file():
            assert b"hf_PLANTED" not in p.read_bytes(), p


def test_default_space_serves_the_runners_default_coordinator():
    """One board URL: the runner's default coordinator is the host of the Space deploy writes to."""
    cli = (paths.runner_dir() / "src" / "aidev_swarm" / "cli.py").read_text()
    runner_default = re.search(r'DEFAULT_COORDINATOR = "([^"]+)"', cli).group(1)
    assert deploy.space_url(deploy.DEFAULT_SPACE) == runner_default == "https://agent-harness-workshop-aidev-swarm.hf.space"
    assert deploy.DEFAULT_SPACE.split("/")[0] == deploy.DEFAULT_BUCKET.split("/")[0], "results live in the Space's org"


class FakeHub:
    def __init__(self, *, bucket_private=True, hardware="cpu-basic"):
        self.calls, self.secrets, self.variables = [], {}, {}
        self.bucket_private, self.hardware = bucket_private, hardware

    def create_bucket(self, bucket, private, exist_ok):
        self.calls.append(("create_bucket", bucket, private))

    def bucket_info(self, bucket):
        return type("B", (), {"private": self.bucket_private})()

    def create_repo(self, space, **kw):
        self.calls.append(("create_repo", space, kw.get("private")))

    def get_space_runtime(self, space):
        return type("R", (), {"requested_hardware": self.hardware, "hardware": self.hardware})()

    def request_space_hardware(self, space, hardware):
        self.calls.append(("hardware", hardware))

    def add_space_secret(self, space, key, value):
        self.secrets[key] = value

    def add_space_variable(self, space, key, value):
        self.variables[key] = value

    def upload_folder(self, **kw):
        self.calls.append(("upload", kw["repo_id"]))
        return type("C", (), {"oid": "abc123"})()


def token_file(tmp_path, value="hf_SCOPEDorgTOKENvalue123"):
    f = tmp_path / "space-token.txt"
    f.write_text(f"# comment naming the token\nSPACE_HF_TOKEN={value}\n")
    return f


def test_deploy_gives_the_space_the_scoped_org_token_from_the_token_file(tmp_path, capsys):
    hub = FakeHub()
    assert deploy.deploy("org/space", "org/results", api=hub, token_file=token_file(tmp_path)) == "abc123"
    assert hub.secrets == {"HF_TOKEN": "hf_SCOPEDorgTOKENvalue123"}
    assert hub.variables == {"SWARM_BUCKET": "org/results"}
    out = capsys.readouterr()
    assert "hf_SCOPED" not in out.out + out.err


@pytest.mark.parametrize("content", [None, "# only comments\n", "SPACE_HF_TOKEN=\n"])
def test_deploy_refuses_without_a_scoped_token_before_any_hub_call(tmp_path, content):
    f = tmp_path / "space-token.txt"
    if content is not None:
        f.write_text(content)
    hub = FakeHub()
    with pytest.raises(SystemExit, match="SPACE_HF_TOKEN"):
        deploy.deploy("org/space", "org/results", api=hub, token_file=f)
    assert hub.calls == [] and hub.secrets == {}


def test_deploy_refuses_a_public_results_bucket(tmp_path):
    hub = FakeHub(bucket_private=False)
    with pytest.raises(SystemExit, match="public"):
        deploy.deploy("org/space", "org/results", api=hub, token_file=token_file(tmp_path))
    assert hub.secrets == {} and ("upload", "org/space") not in hub.calls


def test_deploy_serves_a_public_space_and_creates_a_missing_one_public(tmp_path):
    """Joel, 18 Nov: the Space (board, /join) is public; only the results bucket stays private."""
    hub = FakeHub()  # an existing public Space keeps its visibility: deploy no longer reads it
    assert deploy.deploy("org/space", "org/results", api=hub, token_file=token_file(tmp_path)) == "abc123"
    assert ("create_repo", "org/space", False) in hub.calls and ("create_bucket", "org/results", True) in hub.calls
    assert ("upload", "org/space") in hub.calls


def test_deploy_requests_the_hardware_of_an_existing_space(tmp_path):
    hub = FakeHub(hardware="cpu-basic")
    deploy.deploy("org/space", "org/results", hardware="cpu-upgrade", api=hub, token_file=token_file(tmp_path))
    assert ("hardware", "cpu-upgrade") in hub.calls
    hub = FakeHub(hardware="cpu-basic")
    deploy.deploy("org/space", "org/results", api=hub, token_file=token_file(tmp_path))
    assert not any(c[0] == "hardware" for c in hub.calls)
