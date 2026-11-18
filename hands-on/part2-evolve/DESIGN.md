# Part 2 hands-on: evolve the harness — design

Who decided: **Joel** (his brief or answer), **parent** (the orchestrating chat), **helper** (this build;
tell me if you disagree). Dates 17 Nov 2026.

## What an attendee does

```bash
uvx --from "<wheel URL>" aidev-evolve check                     # HF login, your coding agent, the room board
uvx --from "<wheel URL>" aidev-evolve run --proposer claude     # or codex, or manual; posts to <board>/evolve
uvx --from "<wheel URL>" aidev-evolve replay                     # offline: a recorded real run
```

The terminal shows: the seed agent's scores, then per iteration the proposer's patch title and hypothesis,
the coloured diff of `agent.py`, each trial landing live, the validation score before -> after, KEPT or
DROPPED with the fixed and broken tasks, a bar chart of the best agent so far, and the HF credits spent.
At the end a held-out check compares the seed with the best agent on tasks the loop never saw.

## The loop

1. **Seed calibration** (one HF Job): seed on the search split x1 and the validation split x2. The two
   validation runs give the noise floor (flip rate d) and the keep margin `max(1 task, sqrt(d/n))`.
2. **Propose**: a fresh `work/` folder with `agent.py` (the current best), `traces/` (its transcripts on
   the search tasks, failures first, as readable markdown), `HISTORY.md` (every earlier attempt and its
   verdict), `CONTRACT.md` and `PROMPT.md`. The attendee's `claude -p` / `codex exec` edits `agent.py`
   and writes `PATCH.md` (title, hypothesis, change, expected effect). `manual` prints the prompt path
   and waits for Enter (any coding agent, or the attendee's own hands).
3. **Check before spending**: static audit (stdlib allowlist, no `open`/`eval`, no private attributes,
   no benchmark task names) and a preflight run against a scripted model (catches crashes).
4. **Evaluate** (one HF Job): child on search + validation x1, **parent re-measured on validation x1 in
   the same Job** (paired, interleaved).
5. **Keep or drop**: child's validation pass rate vs the parent's pooled over all its measurements; keep
   if the gain >= margin. A child that wins is measured once more with its parent (one more Job, eval id
   `eNNNr`) and kept only if it still wins on everything pooled. The kept child's search transcripts feed the
   next proposal.
6. **Held-out check** (one HF Job): seed vs best on the held-out split, 2 trials per task; the paired gain with a
   95% CI that resamples tasks (`stats.paired_diff`).

## Decisions

| # | Decision | Why | By |
|---|---|---|---|
| 1 | Same model and benchmark as Part 1: GLM-5.3-Flash (zai-org, effort low) on Terminal-Bench 2.1 | Explain once | Joel |
| 2 | Attendee's own Codex / Claude Code subscription proposes; `manual` for everyone else | BYO, vendor-neutral | Joel |
| 3 | Everything runs on HF in the attendee's account: one cpu-basic HF Job per evaluation, HF Sandboxes per trial, Part 1's pinned `hlb` sandbox environment | Reuse Part 1's proven path; laptop install stays small | Joel (HF), parent (reuse) |
| 4 | **The agent loop runs in the Job, not in the task container** (`EvolveAgent`, a Harbor agent that calls `environment.exec`) | Nothing to install per trial (Part 1's harness installs took 10-150 s), agent.py stays pure Python, the proposer cannot reach the model settings or the recorder | helper |
| 5 | The candidate's source travels inside the Job spec (one env variable); results come back as gzip+base64 lines in the Job log | No bucket and no write token scope needed; probe: 60 KB variable and 16 KB lines fine | helper |
| 6 | **Phase 2 seed (config B)** = 58 lines with four weaknesses: 10-step limit, model sees the first 1,000 chars of each output, stderr hidden (`2>/dev/null`), stops at the first text-only reply (plus no command timeout, a stateless shell) | Sweeps (PROGRESS.md, Phase 2): hung commands, head-only output, hidden stderr and unverified stops cost GLM-5.3-Flash only 4-6 points on the fast tasks; a step limit is the weakness that bites inside the budget. Seed 60% vs a hand-fixed reference 80% on 34 tasks x 4 | helper |
| 7 | Per-trial agent budget **180 s** (was 150 s), enforced by the runtime (Harbor's own timeout 90 s behind); verifier capped at 240 s and not scored when it times out. The splits file carries the budget its tasks were selected for (`budget_s`), the default everywhere | Config A (240 s) held up as well but costs about 25% more and takes longer per iteration | helper |
| 8 | Splits (`data/splits.json`, rule in `devtools/split_tasks.py`): search 4 (seed failures the proposer reads), validation 6 (4 headroom + 2 guards), held-out 8 (4 headroom + 4 guards). Headroom = hand-fixed reference >= 3/4 and >= 2/4 above the seed; guard = both 4/4 | Phase 1's split had no headroom on held-out and noisy validation | helper |
| 9 | Keep rule: paired, parent re-measured each iteration and pooled, validation 2 trials, margin max(1 task, z sqrt(d/(n k))) with z = 1.5, a winning child re-measured once and decided on the pooled data | Phase 1's 8 x 1, z = 1 rule kept a lucky child that did not hold | helper |
| 10 | Infra trials (sandbox, token, credits) never count; agent timeouts count as fails | Same rule as the leaderboard | helper |
| 11 | No spending cap; running spend is displayed every iteration; target: one full run well under $3 | Joel (correction, 17 Nov) | Joel |
| 12 | Room board: client + stand-in here, API proposal in PROTOCOL.md; the coordinator owner integrates | Parent's brief | parent |
| 13 | Offline fallback: `transcript.jsonl` of every run; `replay` plays a bundled real run | Speaker guidelines (Wi-Fi may fail) | parent (requirement), helper (design) |
| 14 | Wheel = release asset of the public GitHub repo (tag `evolve-v<version>`), like Part 1; the Job downloads it without credentials | Attendees need no access to anything private; the HF token never goes to the wheel host | Joel (18 Nov) |
| 15 | Proposer default model for `claude`: whatever the attendee passes (`--model sonnet` used in the Phase 1 run for speed) | Sonnet answers in ~20-60 s | helper |

## Run folder (entry point for Phase 2)

```
runs/<name>/
  config.json        splits, budget, model, proposer
  state.json         incumbent, margin, flip rate, history (one entry per iteration), curve, spend, held-out
  records.jsonl      every trial: candidate, task, split, eval id, outcome, wall, cost, full trajectory
  transcript.jsonl   the terminal output with times (replay source)
  candidates/cNNN/   agent.py, PATCH.md, diff.patch
  work/              the proposer's folder of the last iteration
```

Phase 2 commands: `aidev-evolve experiment --runs R --iterations N --heldout-trials K --run-dir runs/exp-...`
(R independent runs, `summary.json`), `aidev-evolve eval --agent <agent.py> --tasks a,b --trials K`
(re-measure any candidate, e.g. the final agents on held-out with more trials).

## Room-scale risks (not solved in Phase 1)

- **Concurrency.** One iteration = 22 sandboxes at once per attendee. 250 attendees = 5,500 sandboxes and
  as many concurrent GLM conversations; the measured zai-org budget was 393 concurrent trials (Part 1
  notes). Options: fewer trials per iteration (validation 6, search 4), staggered starts, a second
  admitted provider (novita), or most of the room on `replay` + a few live runs. Needs a load test.
- **Sandbox quota** per new account with a coupon: untested (same as Part 1).
- **Proposer limits**: Claude Pro programmatic credit rules unverified; Codex is the cleaner BYO route.

## Phase 1 result and what Phase 2 must change (helper, 17 Nov; details in PROGRESS.md)

The loop works end to end on HF, but the one patch it kept (lift the 15-step limit) did **not** hold in a
2-trial re-measure on all 28 tasks (seed 33/56, c001 31/56). Two causes, each with a fix to test:

1. **Headroom is in the wrong place.** The tasks where the seed fails and good harnesses pass
   (financial-document-processor, build-pov-ray, sparql-university, ...) need more than 150 s, so harness
   fixes cannot convert them inside the budget. Options: (a) budget 240-300 s on those tasks (iteration about
   6-7 min, 3 iterations in 20 min); (b) a different seed weakness that bites inside 150 s and is mechanical
   (for example: no output cap at all so long outputs flood the context, or a tool-call format the model often
   breaks), measured with 4 trials before fixing splits; (c) a cheaper or weaker model than GLM-5.3-Flash
   (breaks decision 1, Joel's call).
2. **Noise.** Per-task pass rates of the seed sit near 50% on many tasks; 8 validation tasks x 1 trial and
   z = 1 accept noise. Use validation tasks with stable seed outcomes (0/4 or 4/4 in the sweep), `--val-trials 2`
   (cost about +60% per iteration), and a margin with z >= 1.5; re-measure a kept child once before it becomes
   the parent.
3. **Held-out split had no headroom** (12 of 14 tasks the seed passes 2/2). Rebuild it from the sweep with the
   same mix as validation, 2+ trials.
4. Replace `headless-terminal` (verifier hangs, 6-7 min iterations) with a reserve task.

Entry points: `aidev-evolve eval` (sweeps, re-measures), `aidev-evolve experiment` (R runs + held-out,
`summary.json`), `runs/sweep-seed-summary.txt` (seed per-task rates and the best board harness's).

## Phase 2 result (helper, 17 Nov; details and numbers in PROGRESS.md)

Two configurations, two real loops of 8 iterations each, held-out re-checks, and a check on every task the loop
never saw. **Config B is the default** (decisions 6-9 above): its gain holds on all 24 unseen tasks (+15 and +17
points, 95% CI over tasks [+2, +31]); config A (Phase 1 seed, 240 s) looked as good on its 8 held-out tasks but
scored 2-4 points below its seed on the 25 unseen tasks. Open for the room: after the iteration-1 fix, validation
saturates at 83-90% and later patches cannot clear the margin, so attendees see one big kept patch followed by
dropped ones (gallery: talk folder `evolve/patch-gallery.html`).

## Phase 3: attendees own an optimization strategy, and the room ranks strategies (helper, 18 Nov)

Joel: "let them make some changes to the optimization process so we have a winner in the end and not just random
noise". Each attendee runs `aidev-evolve run --strategy <preset|folder>`; the board ranks strategies by held-out gain.

**Strategy = `strategy.toml` + `proposer_prompt.md`** (`src/aidev_evolve/strategy.py`, presets in
`src/aidev_evolve/strategies/`; `aidev-evolve strategy-init small-diffs my-strategy` copies one in a second):

| Knob | Values | What it changes |
|---|---|---|
| proposer prompt (`proposer_prompt.md`) | free text, <= 4,000 chars | how the coding agent approaches a patch; the loop appends fixed rules (contract, imports, no task names, PATCH.md) |
| `evidence.traces` | `failed` / `all` | which search transcripts it reads (failures first) |
| `evidence.max_traces` | 1-8 | how many |
| `evidence.history` | `none` / `titles` / `full` | what it sees of earlier attempts |
| `evidence.diffs` | bool | earlier attempts' diffs in HISTORY.md |
| `proposer.candidates` | 1 / 2 | two proposals per iteration, both measured in one Job with the parent, the better goes through the keep rule (re-measure forced: picking the better of two is a selection) |
| `proposer.second_hint` | text | steers the second proposal (preset: "fix the second most common failure") |
| `keep.val_trials`, `keep.z`, `keep.confirm` | 1-3, 0-3, bool | the keep rule |
| `iterations` | 1-6 | the most iterations; the session time box decides how many start |

**Fixed (refused in a strategy file):** model and effort, 180 s per task, the splits and held-out set, the seed, 2
held-out trials, the 22-min session (an iteration starts only if it, as long as the longest so far, and the 5.5-min
held-out check still fit), and the $2 worst-case credit cap (checked when the strategy loads). The proposer (Claude
Code or Codex, and its model) is the attendee's own and is recorded, not fixed.

**Presets:** `baseline` (Phase 2 loop: all transcripts, full history, 1 candidate, z 1.5), `failures-first` (only
failures, a FAILURES.md tally first, titles-only history), `small-diffs` (<= 10 changed lines, z 1.0, sees earlier
diffs), `two-candidates` (2 per iteration, 3 iterations max). Each fits the session with 2-3 iterations and costs at
most $1.4-1.9 on splits C.

**Winner, not noise (`src/aidev_evolve/ranking.py`):** every run ends with seed vs final on the 23 held-out tasks x 2
trials (about 5 min, $0.35). One run's gain has a ±17-point CI, so individuals are not ranked. Runs are pooled per
strategy (identical knobs + prompt = same hash; a by-preset view pools edits with their preset); score = mean final
pass rate; gain = score - seed pooled over the room; CI = Welch t over run scores (tasks fixed); ties = CI of the
difference covers 0; tie groups and shared ranks with the Harness Leaderboard rule; < 5 runs = listed, not ranked.
Final view: winning strategy (or "tie for first") and winning patch (best run in that group with its own CI).
Numbers: talk folder `evolve/strategy-noise.md`.

**Result (19 real runs, PROGRESS.md):** the machinery works (every run ends with a paired held-out gain; ties are
called), but the five strategies tried tie: every run keeps the same first patch (lift the step limit, +15-25 points
held-out) and a second patch was kept once in 19 runs (it did not hold on held-out). The kill fix was proposed once in
41 proposals. Likely cause: the search transcripts show few killed commands. Next step (open): kill-heavy tasks in search.

| # | Decision | Why | By |
|---|---|---|---|
| 16 | Seed C = seed B4 + every command killed after 10 s; splits C (validation 7, held-out = the 23 other pool tasks) | Sweep: step fix +22, kill fix +13, residual after the step fix +10 (B: +3.7); held-out large enough to rank strategies | helper (gate from the runbook) |
| 17 | Rank strategies (pooled runs), not individuals; Welch t over runs, tasks fixed; min 5 runs | Simulation: one run ±17 pts; bootstrap over 5 runs under-covers (11% false winners), Welch 4% | helper |
| 18 | Session time box 22 min (not 20) | Calibration 3 + 2 iterations 8.4 + one re-measure 2.5 + held-out 5.5 = 19.4 min; 20 would allow only 1-2 iterations | helper |
| 19 | Held-out trials 2 per task (not 1) | 1 trial halves the power of the strategy comparison (gap 8, 5 runs: 40% vs 75%) for $0.18 less | helper |

## Phase 4: a seed whose weaknesses separate strategies (helper, 19 Nov; numbers in PROGRESS.md)

Phase 3's tie had a measurable cause: after the step-limit fix, seed C had nothing left to find. A paired sweep (34
tasks x 4 trials) of the step-fixed harness with and without the 10-s kill gave -0.7 points [-8.5, +7.4]: with unlimited
steps GLM-5.3-Flash works around the kill (backgrounds builds, retries). Moving kill tasks into search could not help.
**Seed E** adds two weaknesses that do not depend on the step limit, each measured with the other two fixed: every
process a command leaves running is killed after it (servers die after the model checked them: -8.8 points [-21.3, +2.9] on the pool; no
transcript shows a failure, only the cleanup appended to each command) and the task text is cut at 1,000 characters
(-19.9 [-33.5, -7.4]). The digest now flags a cut task on step 1. Splits E (`devtools/splits_e.py`): validation holds 2+
tasks per weakness, held-out 21 tasks keeps tasks for each.

20 real runs on seed E: **code-audit** (read agent.py line by line, then the transcripts; fix every harmful line in one
patch) found all three fixes in 5 of 5 runs and ranks first alone (held-out 70.5% vs every-weakness 57.8% and
baseline 50.0%; differences +12.7 [+5.2, +20.1] and +20.5 [+14.0, +26.9], Welch over runs). Transcript-driven
strategies never found the stray-process kill (baseline 0 of 8 kept it, every-weakness 1 of 7). It is the default.

| # | Decision | Why | By |
|---|---|---|---|
| 20 | Seed E = seed C + stray processes killed after every command + task text cut at 1,000 chars; splits E (search 5, validation 8, held-out 21) | Kill fix after the step fix = -0.7 [-8.5, +7.4]: no second weakness left in seed C; the two new ones bite with the rest fixed (-8.8, -19.9 pts); the held-out ranking confirms the third (+12.7 for the strategy that finds it) | helper (parent approved re-splitting; the seed change follows the skill's rule) |
| 21 | Default strategy `code-audit`; presets `every-weakness` and `code-audit` added | Ranked first alone on 20 runs (5/7/8 per strategy) | helper |
| 22 | One GLM provider per attendee (Part 1's list and hash), per-call fallback on 429/5xx/timeout, Job moves after 3 fallbacks; `provider`/`provider_calls` on events | Coordinator request (REQUESTS.md); hash on the username cannot bias the strategy ranking | coordinator owner (request), helper (build) |
| 23 | Session stays 22 min with 2 iterations (3 do not fit: calibration 3 + 2 x 4.5 + re-measure 2.5 + held-out 5.5 = 20 min) | The winner needs one iteration; a third adds 5 min per attendee for nothing measured | helper |
