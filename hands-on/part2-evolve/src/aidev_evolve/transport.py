"""Records from the HF Job to the laptop, as marked lines in the Job's log (no bucket, no extra token scope).

A record is gzip-compressed JSON, base64-encoded and cut into chunks; each chunk is one log line
`@@AIDEV@@ <msg_id> <i> <n> <chunk>`. The decoder reassembles messages, tolerates log prefixes and
returns each message once even when a log reconnect repeats lines. A final `@@AIDEV-DONE@@ <n>` line
says how many records the Job emitted, so the laptop can tell a complete evaluation from a crashed Job.
Probe (17 Nov): 30 lines of 16 KB came back intact through `fetch_job_logs`.
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any

MARK = "@@AIDEV@@"
DONE = "@@AIDEV-DONE@@"


def encode(obj: Any, *, msg_id: str, chunk_chars: int = 12_000) -> list[str]:
    blob = base64.b64encode(gzip.compress(json.dumps(obj, separators=(",", ":")).encode())).decode()
    parts = [blob[i:i + chunk_chars] for i in range(0, len(blob), chunk_chars)] or [""]
    return [f"{MARK} {msg_id} {i} {len(parts)} {p}" for i, p in enumerate(parts)]


class Decoder:
    def __init__(self) -> None:
        self._parts: dict[str, dict[int, str]] = {}
        self._done: set[str] = set()

    def feed(self, line: str) -> list[Any]:
        at = line.find(MARK + " ")
        if at < 0:
            return []
        fields = line[at:].strip().split(" ")
        if len(fields) != 5:
            return []
        _, msg_id, i, n, chunk = fields
        if msg_id in self._done:
            return []
        parts = self._parts.setdefault(msg_id, {})
        parts[int(i)] = chunk
        if len(parts) < int(n):
            return []
        self._done.add(msg_id)
        del self._parts[msg_id]
        blob = "".join(parts[k] for k in range(int(n)))
        return [json.loads(gzip.decompress(base64.b64decode(blob)))]


def done_line(n_records: int) -> str:
    return f"{DONE} {n_records}"


def is_done(line: str) -> bool:
    return (DONE + " ") in line


def done_count(line: str) -> int:
    return int(line[line.index(DONE) + len(DONE):].split()[0])
