# Sample requests for your coding agent (Part 1)

Use these only if you are stuck. Each one is a complete request: copy it into your coding agent, or change it first.
The finished page that these requests lead to is `sample-solution.html` in this folder.

## Challenge 1: the page

```
Build index.html from sample-results.json: one table with a row per harness and the columns harness, pass rate
as "80% (4/5)", trials, total cost with 4 decimals, and a badge "mock" or "measured". Sort by pass rate, highest
first, ties by lower cost. Add one line above the table: "Mock rows are made-up sample data."
```

For the sample data the table shows:

| Harness | Pass rate | Trials | Cost | Data |
|---|---|---|---|---|
| pi | 80% (4/5) | 5 | $0.0200 | mock |
| qwen-code | 60% (3/5) | 5 | $0.0350 | mock |
| openclaw | 20% (1/5) | 5 | $0.0300 | mock |

## Challenge 2: online

```
Make index.html read my measured results from my bucket, once a minute and with a Refresh button, and show the
time of the last update. Then put the page online as a static Space named harness-leaderboard and give me the address.
```

## Challenge 4: more on the page

```
Add a column "Cost per pass" (cost divided by passed tasks, "n/a" when nothing passed). Below the table add a grid:
one row per task, one column per harness, each cell "pass", "fail" or "-". Mark the column of a mock harness with
"(mock)". If the page is online, upload it again.
```

For the sample data: cost per pass is $0.0050 for pi, $0.0117 for qwen-code and $0.0300 for openclaw.

## The finished page

`sample-solution.html` is the result of the three requests above. To use it, copy it to `../index.html`, put your
Hugging Face user name into the line `const HF_USER = "YOUR_USER_NAME";` and ask your coding agent to put it online.
