"""Durable storage of results in a private HF bucket, so a restarted Space rebuilds its SQLite.

Layout: `results/<cell_id>.json` (one per result, as stored), `state/leases.json` (the live
leases) and `evolve/<user>/<run_id>/<kind>-<iteration>.json` (Part 2 board events, `put_path`). Writes are batched: `put` only queues, `flush` sends one `batch_bucket_files` call
(a background thread flushes every few seconds). A failed flush keeps everything queued.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("swarm.sink")

LEASES = "state/leases.json"


class NullSink:
    def put(self, result: dict[str, Any]) -> None:
        pass

    def put_leases(self, leases: list[dict[str, Any]]) -> None:
        pass

    def put_path(self, path: str, data: bytes) -> None:
        pass

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return [], []

    def load_prefix(self, prefix: str) -> dict[str, bytes]:
        return {}

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


class BucketSink:
    def __init__(self, api: Any, bucket_id: str, *, interval_s: float = 5.0):
        self.api, self.bucket_id, self.interval_s = api, bucket_id, interval_s
        self._pending: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def put(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._pending[f"results/{result['cell_id']}.json"] = json.dumps(result, separators=(",", ":")).encode()

    def put_leases(self, leases: list[dict[str, Any]]) -> None:
        with self._lock:
            self._pending[LEASES] = json.dumps(leases, separators=(",", ":")).encode()

    def put_path(self, path: str, data: bytes) -> None:
        with self._lock:
            self._pending[path] = data

    def flush(self) -> bool:
        with self._lock:
            batch, self._pending = self._pending, {}
        if not batch:
            return True
        try:
            self.api.batch_bucket_files(self.bucket_id, add=[(data, path) for path, data in batch.items()])
            return True
        except Exception as e:  # keep the batch; newer versions of the same path win
            log.warning("bucket flush failed (%s); %d files stay queued", type(e).__name__, len(batch))
            with self._lock:
                self._pending = {**batch, **self._pending}
            return False

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results = [json.loads(d) for p, d in sorted(self.load_prefix("results/").items()) if p.endswith(".json")]
        leases = self.load_prefix(LEASES).get(LEASES)
        return results, json.loads(leases) if leases else []

    def load_prefix(self, prefix: str) -> dict[str, bytes]:
        return {}

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


class BucketSink:
    def __init__(self, api: Any, bucket_id: str, *, interval_s: float = 5.0):
        self.api, self.bucket_id, self.interval_s = api, bucket_id, interval_s
        self._pending: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def put(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._pending[f"results/{result['cell_id']}.json"] = json.dumps(result, separators=(",", ":")).encode()

    def put_leases(self, leases: list[dict[str, Any]]) -> None:
        with self._lock:
            self._pending[LEASES] = json.dumps(leases, separators=(",", ":")).encode()

    def put_path(self, path: str, data: bytes) -> None:
        with self._lock:
            self._pending[path] = data

    def flush(self) -> bool:
        with self._lock:
            batch, self._pending = self._pending, {}
        if not batch:
            return True
        try:
            self.api.batch_bucket_files(self.bucket_id, add=[(data, path) for path, data in batch.items()])
            return True
        except Exception as e:  # keep the batch; newer versions of the same path win
            log.warning("bucket flush failed (%s); %d files stay queued", type(e).__name__, len(batch))
            with self._lock:
                self._pending = {**batch, **self._pending}
            return False

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        paths = [e.path for e in self.api.list_bucket_tree(self.bucket_id, prefix="results/", recursive=True)
                 if getattr(e, "type", "file") == "file" and e.path.endswith(".json")]
        results, leases = [], []
        with tempfile.TemporaryDirectory() as tmp:
            want = [*paths, LEASES]
            for i in range(0, len(want), 200):
                self.api.download_bucket_files(self.bucket_id, [(p, str(Path(tmp) / p)) for p in want[i:i + 200]], raise_on_missing_files=False)
            for p in paths:
                f = Path(tmp) / p
                if f.exists():
                    results.append(json.loads(f.read_text()))
            if (Path(tmp) / LEASES).exists():
                leases = json.loads((Path(tmp) / LEASES).read_text())
        return results, leases

    def load_prefix(self, prefix: str) -> dict[str, bytes]:
        """Every file under `prefix`, as {path: bytes}."""
        paths = [e.path for e in self.api.list_bucket_tree(self.bucket_id, prefix=prefix, recursive=True) if getattr(e, "type", "file") == "file"]
        out: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(0, len(paths), 200):
                self.api.download_bucket_files(self.bucket_id, [(p, str(Path(tmp) / p)) for p in paths[i:i + 200]], raise_on_missing_files=False)
            for p in paths:
                f = Path(tmp) / p
                if f.exists():
                    out[p] = f.read_bytes()
        return out

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                self.flush()
        self._thread = threading.Thread(target=loop, name="bucket-flush", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 1)
        self.flush()
