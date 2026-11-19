"""swarm_coordinator.reset: empty the STAGING board and bucket after a dry run; the prod board is never touched."""

from __future__ import annotations

import pytest

from swarm_coordinator import deploy, reset

STAGING_SPACE, STAGING_BUCKET = deploy.TARGETS["staging"]


class Entry:
    def __init__(self, path, type_="file"):
        self.path, self.type = path, type_


class FakeHub:
    """The Hub calls reset makes; refuses anything on the prod Space or bucket outright."""

    def __init__(self, files, *, stage="RUNNING", drop_deletes=False):
        self.files, self.stage, self.drop_deletes, self.calls = set(files), stage, drop_deletes, []

    def _guard(self, repo):
        assert repo not in deploy.TARGETS["prod"], f"touched prod: {repo}"

    def list_bucket_tree(self, bucket, prefix=None, recursive=None):
        self._guard(bucket)
        return [Entry("results", "directory")] + [Entry(f) for f in sorted(self.files)]

    def batch_bucket_files(self, bucket, delete=None, **kw):
        self._guard(bucket)
        assert self.stage == "PAUSED", "deleted while the Space could still write (it flushes every 5 s)"
        self.calls.append(("delete", len(delete)))
        if not self.drop_deletes:
            self.files -= set(delete)

    def pause_space(self, space):
        self._guard(space)
        self.calls.append(("pause", space))
        self.stage = "PAUSED"

    def restart_space(self, space):
        self._guard(space)
        self.calls.append(("restart", space))
        self.stage = "RUNNING"

    def get_space_runtime(self, space):
        self._guard(space)
        return type("R", (), {"stage": self.stage})()


FILES = ["state/leases.json", "results/pi__fix-git__0.json", "evolve/alice/alice-r1/seed-000.json"]


def empty_board():
    return {"done": 0, "evolve_runs": 0}


def test_reset_pauses_deletes_every_file_restarts_and_checks_the_board_is_empty():
    hub = FakeHub(FILES)
    out = reset.reset(hub, board=empty_board, sleep=lambda s: None)
    assert hub.files == set() and hub.calls[0] == ("pause", STAGING_SPACE) and hub.calls[-1] == ("restart", STAGING_SPACE)
    assert out == {"deleted": 3, "board": {"done": 0, "evolve_runs": 0}, "stage": "RUNNING"}


def test_reset_can_leave_staging_paused():
    hub = FakeHub(FILES)
    out = reset.reset(hub, board=empty_board, sleep=lambda s: None, pause=True)
    assert hub.files == set() and ("restart", STAGING_SPACE) not in hub.calls and out["stage"] == "PAUSED"


def test_reset_deletes_in_batches():
    hub = FakeHub([f"results/c{i}.json" for i in range(1201)])
    reset.reset(hub, board=empty_board, sleep=lambda s: None)
    assert [c for c in hub.calls if c[0] == "delete"] == [("delete", 500), ("delete", 500), ("delete", 201)] and not hub.files


def test_reset_refuses_when_files_survive_the_delete():
    hub = FakeHub(FILES, drop_deletes=True)
    with pytest.raises(SystemExit, match="still"):
        reset.reset(hub, board=empty_board, sleep=lambda s: None)
    assert ("restart", STAGING_SPACE) not in hub.calls


def test_reset_fails_when_the_restarted_board_is_not_empty():
    hub = FakeHub(FILES)
    with pytest.raises(SystemExit, match="not empty"):
        reset.reset(hub, board=lambda: {"done": 2, "evolve_runs": 0}, sleep=lambda s: None)


@pytest.mark.parametrize("space,bucket", [deploy.TARGETS["prod"], (STAGING_SPACE, deploy.DEFAULT_BUCKET), (deploy.DEFAULT_SPACE, STAGING_BUCKET)])
def test_reset_never_touches_prod(space, bucket):
    hub = FakeHub(FILES)
    with pytest.raises(SystemExit, match="prod"):
        reset.reset(hub, board=empty_board, sleep=lambda s: None, space=space, bucket=bucket)
    assert hub.calls == []


def test_cli_has_no_prod_target():
    with pytest.raises(SystemExit):
        reset.main(["--target", "prod"])
