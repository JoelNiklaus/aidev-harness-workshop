from __future__ import annotations

import json
from pathlib import Path

from conftest import make_result
from swarm_coordinator.sink import BucketSink


class FakeBucketApi:
    """The slice of HfApi the sink uses, with an in-memory bucket. Fails on demand."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.batches = 0
        self.fail_next = 0

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None, token=None):
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("HTTP 500 from hub")
        self.batches += 1
        for data, path in add or []:
            self.files[path] = data

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        class F:
            def __init__(self, path):
                self.path, self.type = path, "file"
        return [F(p) for p in sorted(self.files) if p.startswith(prefix or "")]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        for path, local in files:
            path = getattr(path, "path", path)
            if path in self.files:
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                Path(local).write_bytes(self.files[path])


def test_flush_writes_pending_results_and_leases_in_one_batch():
    api = FakeBucketApi()
    s = BucketSink(api, "agent-harness-workshop/aidev-swarm-results")
    s.put(make_result("pi__t1__0"))
    s.put(make_result("pi__t2__0"))
    s.put_leases([{"cell_id": "pi__t1__1", "user": "bob", "expires": 2e9}])
    s.flush()
    assert api.batches == 1
    assert set(api.files) == {"results/pi__t1__0.json", "results/pi__t2__0.json", "state/leases.json"}
    s.flush()
    assert api.batches == 1  # nothing pending: no call


def test_failed_flush_keeps_pending_for_the_next_one():
    api = FakeBucketApi()
    api.fail_next = 1
    s = BucketSink(api, "b")
    s.put(make_result("pi__t1__0"))
    s.flush()
    assert api.files == {}
    s.flush()
    assert "results/pi__t1__0.json" in api.files


def test_load_reads_back_results_and_leases():
    api = FakeBucketApi()
    s = BucketSink(api, "b")
    s.put(make_result("pi__t1__0", user="alice"))
    s.put_leases([{"cell_id": "pi__t2__0", "user": "bob", "expires": 2e9}])
    s.flush()
    results, leases = BucketSink(api, "b").load()
    assert [r["hf_user"] for r in results] == ["alice"]
    assert leases == [{"cell_id": "pi__t2__0", "user": "bob", "expires": 2e9}]
    assert json.loads(api.files["results/pi__t1__0.json"])["cell_id"] == "pi__t1__0"


def test_load_of_an_empty_bucket():
    assert BucketSink(FakeBucketApi(), "b").load() == ([], [])


def test_close_flushes_what_is_still_queued():
    api = FakeBucketApi()
    s = BucketSink(api, "b", interval_s=3600)
    s.start()
    s.put(make_result("pi__t1__0"))
    s.close()
    assert "results/pi__t1__0.json" in api.files
