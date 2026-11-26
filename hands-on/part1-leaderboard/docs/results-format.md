# Results format

`results.json` is a JSON list of records. One record is one trial: one harness on one task. `aidev-lb run`
writes the file in the current folder and, as a copy, in the attendee's public bucket:

`https://huggingface.co/buckets/USER/harness-results/resolve/results.json`

A new result of the same harness and task replaces the old one. A trial that ended without a result (an
infrastructure error, or the 6-minute limit) has no record.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `cell_id` | string | `<harness>__<task>__<trial>`, unique in the list |
| `harness` | string | harness name, as in `aidev-lb harnesses` |
| `harness_version` | string | version of the harness that ran |
| `task` | string | Terminal-Bench 2.1 task name |
| `trial` | integer | always 0 in the workshop |
| `model` | string | `zai-org/GLM-5.3-Flash` |
| `reward` | number | 1.0 or 0.0 from the task's tests |
| `passed` | boolean | `reward` is 1.0 |
| `started_at`, `finished_at` | string | UTC time stamps |
| `wall_s` | number | seconds from sandbox start to the end of the tests |
| `n_input_tokens`, `n_cached_tokens`, `n_output_tokens` | integer | tokens of all model calls of the trial |
| `n_calls` | integer | number of model calls |
| `cost_usd` | number | model cost at list price |
| `error` | string or null | error of a trial that still got a score |
| `hf_user` | string | account that ran the trial |
| `runner_version` | string | version of `aidev-lb` |
| `sandbox_id` | string or null | id of the trial's sandbox |
| `trajectory_url` | string or null | not set in the workshop |
| `provider` | string | inference provider that served the model calls |
| `n_provider_errors` | integer | model calls the provider refused or rate-limited |
| `source` | string | `sample` or `measured` |

## `source` values

`sample`: made up, shipped in `sample-results.json`. The page shows these as mock rows.

```json
{"cell_id": "pi__kv-store-grpc__0", "harness": "pi", "harness_version": "0.0.0", "task": "kv-store-grpc",
 "trial": 0, "model": "zai-org/GLM-5.3-Flash", "reward": 1.0, "passed": true,
 "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:01:00Z", "wall_s": 60.0,
 "n_input_tokens": 40000, "n_cached_tokens": 20000, "n_output_tokens": 400, "n_calls": 4, "cost_usd": 0.002,
 "error": null, "hf_user": "sample", "runner_version": "sample", "sandbox_id": null, "trajectory_url": null,
 "provider": "sample", "n_provider_errors": 0, "source": "sample"}
```

`measured`: a trial the attendee ran with `aidev-lb run`.

```json
{"cell_id": "pi__log-summary-date-ranges__0", "harness": "pi", "harness_version": "1.2.0",
 "task": "log-summary-date-ranges", "trial": 0, "model": "zai-org/GLM-5.3-Flash", "reward": 1.0, "passed": true,
 "started_at": "2026-11-25T12:57:12.052879Z", "finished_at": "2026-11-25T12:57:37.571374Z", "wall_s": 25.518,
 "n_input_tokens": 6504, "n_cached_tokens": 3904, "n_output_tokens": 529, "n_calls": 3, "cost_usd": 0.00077162,
 "error": null, "hf_user": "alice", "runner_version": "0.5.0", "sandbox_id": "6b06db28d91f621b9307acb0",
 "trajectory_url": null, "provider": "together", "n_provider_errors": 0, "source": "measured"}
```

The bucket also holds one file per trial, `results/<cell_id>.json`, written by the cloud job. The page reads
only `results.json`.
