"""The results store: where a Job's finished trials go and where the laptop reads them.

Backend "log" (always on): the Job has no storage of its own, so `write_result` there prints the Result as one event
line on the Job log; the laptop's `run` reads that line from the log stream and calls `write_result` again with
`local`, which keeps results.json. `read_all` reads that file.

Backend "bucket" (the default; off with --no-bucket or when the token cannot write it): a public bucket in the attendee's namespace, which a static page can
read without a token. Each Job also writes one file per cell (results/<cell_id>.json), so parallel Jobs never
overwrite each other; the aggregate results.json has one writer, the laptop (`publish`, a copy of the local file).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aidev_lb import events

LOCAL = Path("results.json")
AGGREGATE = "results.json"  # in the bucket: every Result, the same list as the local file
BUCKET = "harness-results"  # the default bucket name, in the attendee's namespace
Upload = Callable[..., None]


def _upload(token: str, bucket: str, files: list[tuple[bytes, str]], *, create: bool = False) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # upload bars would cut through the trial lines
    api = HfApi(token=token)
    if create:
        api.create_bucket(bucket, private=False, exist_ok=True)
    api.batch_bucket_files(bucket, add=files)


def public_url(bucket: str) -> str:
    """Where a page reads the aggregate, without a token."""
    return f"https://huggingface.co/buckets/{bucket}/resolve/{AGGREGATE}"


class BadResultsFile(ValueError):
    """The local file is not a list of Results."""


def read_all(local: Path = LOCAL) -> list[dict[str, Any]]:
    """Every Result of the local file, oldest first; empty when there is none yet."""
    if not Path(local).exists():
        return []
    try:
        results = json.loads(Path(local).read_text())
    except ValueError:
        results = None
    if not isinstance(results, list) or not all(isinstance(r, dict) and {"cell_id", "harness", "task", "source"} <= set(r) for r in results):
        raise BadResultsFile(f"{local} is not a list of results (each with cell_id, harness, task, source). Move it away or name another file with --out.")
    return results


def write_result(result: dict[str, Any], *, local: Path | None = None, out: Callable[[str], None] = print,
                 bucket: str | None = None, token: str | None = None, upload: Upload = _upload) -> None:
    """In the Job (`local` None): hand the Result over as one event line, and with a `bucket` also as the cell's own
    file there (an error of that write reaches the caller after the line is out). On the laptop: put it into `local`,
    one Result per cell (a new one replaces the old)."""
    if local is None:
        out(events.line("result", result=result))
        if bucket:
            upload(token, bucket, [(json.dumps(result).encode(), f"results/{result['cell_id']}.json")])
        return
    local = Path(local)
    kept = read_all(local)
    at = next((i for i, r in enumerate(kept) if r["cell_id"] == result["cell_id"]), None)
    if at is None:
        kept.append(result)
    else:
        kept[at] = result
    tmp = local.with_name(local.name + ".tmp")
    tmp.write_text(json.dumps(kept, indent=1) + "\n")
    os.replace(tmp, local)


def publish(local: Path, bucket: str, token: str, *, upload: Upload = _upload) -> None:
    """Laptop only: write the local results as the bucket's aggregate file, creating the public bucket if needed."""
    upload(token, bucket, [(json.dumps(read_all(local)).encode(), AGGREGATE)], create=True)
