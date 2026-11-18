Read `traces/INDEX.md` and every failed transcript in `traces/`, then `HISTORY.md` (titles and verdicts of
earlier attempts), `CONTRACT.md` and `agent.py`.

Before you edit anything, write `FAILURES.md`: one line per failed transcript naming the mechanism that made
it fail (for example: ran out of steps, a command hung, the output it needed was cut, it stopped without
checking, it misread an error). Count how often each mechanism appears.

Then fix the most frequent mechanism that no earlier iteration has already tried, with one focused change to
`agent.py`. If the top mechanism was tried and dropped, take the next one.
