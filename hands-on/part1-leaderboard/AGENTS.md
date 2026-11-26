# Harness leaderboard: facts for the coding agent

You help a workshop attendee build their own leaderboard page. It compares coding-agent harnesses that all use the
same model on the same Terminal-Bench 2.1 tasks. The human decides what the page shows and how it looks. They ask in
their own words, often in one short sentence. Build what they ask for. Where they leave something open, choose the
simplest thing that works and say what you chose. The hands-on time is 25 minutes: finish a request in 1 to 3 minutes.

## The data

A **record** is one trial: one harness on one task. The fields a page needs (all fields: `docs/results-format.md`):

| Field | Type | Meaning |
|---|---|---|
| `harness` | string | for example `pi` |
| `task` | string | for example `kv-store-grpc` |
| `passed` | boolean | the task's tests passed |
| `cost_usd` | number | model cost of the trial in US dollars |
| `wall_s` | number | seconds of the trial |
| `source` | string | `sample` (made up) or `measured` (the attendee's own run) |

Two lists of records exist:

- **Sample**: `sample-results.json` in this folder. 15 made-up records (3 harnesses x 5 tasks), all with
  `source: "sample"`. A browser does not let a page opened from disk fetch a local file, so bundle the records
  inside the page.
- **Measured**: `https://huggingface.co/buckets/USER/harness-results/resolve/results.json`, a file in the attendee's
  public Hugging Face bucket (`USER` from `hf auth whoami`). `aidev-lb run` writes it, and a copy as `results.json`
  in this folder. The bucket file does not exist before the first run (HTTP 401 or 404). It holds only the
  harnesses the attendee ran, and these can be harnesses that are not in the sample.

## What every page must do

However the page looks, these hold:

1. **The mock rule.** Per harness: if the measured list has at least one record of a harness, the page uses
   only the measured records of that harness. Otherwise it uses its sample records and shows the word `mock` next to that
   harness, in every view (table, chart, grid). Never mix sample and measured records inside one harness. A measured
   harness that is not in the sample appears as well.
2. **It opens from disk** by double-click and works as a static site: one file `index.html` with plain HTML, CSS and
   JavaScript, no build step, no external scripts, fonts or images. Use something else only if the human asks for it.
3. **Reading the measured results** (as soon as the page reads them): `fetch` the address above with
   `cache: "no-store"`, once at the start and then at most every 60 seconds, not while the tab is hidden, plus a
   button that reads now. All attendees share one network address and its read limit. On any status other than 200
   or on a network error, keep what the page shows. No error popup.
4. Write values into the page with `textContent`, or escape them.

## Putting the page online

A static Hugging Face Space. The human chooses the name; without a choice, use `harness-leaderboard`.
An online page has to read the measured results (point 3 above): add that first if the page does not do it yet,
with the user name from `hf auth whoami` in the address.

```bash
hf repos create USER/NAME --repo-type space --space-sdk static --exist-ok
hf upload USER/NAME index.html index.html --repo-type space
```

The page is then at `https://USER-NAME.static.hf.space/` (user name in lower case, `_` and `.` become `-`). It can
take 10 seconds to appear. When you put the page online, also write the line `<!-- space: USER/NAME -->` into
`index.html`. After every later change to an `index.html` that has this line, upload the file to that Space again
yourself and say so.

## Guard rails

- Work only in this folder. Do not edit `sample-results.json` or `docs/`.
- Do not open `hints/` unless the human asks for a hint or a sample.
- Do not run `aidev-lb run` unless the human asks for it. It starts cloud jobs that cost credits.
  `aidev-lb harnesses` lists the 10 harnesses with a speed label and costs nothing.
- Never print, read or store the Hugging Face token. The `hf` and `aidev-lb` commands find it themselves.
- If a command fails, read its message, try the fix it names once, then tell the human. Do not retry in a loop.
  A `401` or `403` from `hf repos create` or `hf upload` means that the token cannot write: the human creates a token
  of type "Write" at https://huggingface.co/settings/tokens and runs `hf auth login` again.
- Before you report, check your work without a browser window: the file exists, the 15 records are in it, and the
  mock rule is in the code. Then report in at most 5 lines: what you built, what you chose, and how to open it.
