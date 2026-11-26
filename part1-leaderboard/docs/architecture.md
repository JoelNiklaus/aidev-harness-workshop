# What runs where

```
laptop                         Hugging Face (the attendee's account)
------                         -------------------------------------
aidev-lb run --harness pi ---> cloud job (HF Job, cpu-basic), one per harness
   ^   reads the job log         Harbor starts 5 trials at the same time
   |                               trial = one sandbox (HF Sandbox)
   |                                 harness (for example Pi) works on the task
   |                                 meter: counts tokens, holds the token
   |                                   -> router -> inference provider (GLM-5.3-Flash)
   |                                 the task's tests give pass or fail
   +--- one line per trial   <--- result
results.json (local)  ------> bucket USER/harness-results (public): results.json
                                        ^
index.html --- hf upload ---> static Space USER/harness-leaderboard: reads results.json every 60 s
```

| Part | Where | What it does |
|---|---|---|
| `aidev-lb` | laptop | Starts the jobs, prints one line per trial, writes `results.json` locally and to the bucket. Small install: no Harbor on the laptop. |
| Cloud job | Hugging Face Jobs | One per harness. Installs the runner with Harbor and runs all trials of its harness in parallel. |
| Harbor | inside the job | The evaluation framework: creates a sandbox per trial, installs the harness, runs the agent, runs the task's tests. |
| Sandbox | Hugging Face Sandbox | One per trial. A fresh container with the task's files. Removed after the trial. |
| Meter | inside the sandbox | A small proxy between the harness and the model. It counts tokens and cost. The harness gets a dummy key, so only the meter holds the attendee's token. |
| Router | Hugging Face Inference Providers | Sends each model call to an inference provider. Each attendee has a fixed first provider, picked from the user name, and falls back to the next when it refuses a call. |
| Bucket | Hugging Face bucket, public | Holds `results.json`. A browser can read it without a token. |
| Space | Hugging Face Space, static | Serves `index.html`. No server code. |

The attendee's account pays for the jobs, the sandboxes and the model calls. `aidev-lb run` ends when every
trial has a result. After 15 minutes it stops waiting and cancels the jobs and sandboxes still running.

## Why the page polls only every 60 seconds

Reads without a token are limited per network address: 3,000 requests per 5 minutes. A conference room shares
few addresses. So the page reads once per minute, skips the read when its tab is hidden, and has a Refresh
button.
