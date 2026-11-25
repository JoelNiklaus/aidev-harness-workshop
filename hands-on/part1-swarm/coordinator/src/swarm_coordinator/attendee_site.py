"""The attendee site: static pages for GitHub Pages (the public repo's docs/ folder), opened from the slide's QR code.

    python3 -m swarm_coordinator.attendee_site <public checkout>/docs      # tools/export_public.sh runs this

/1 is Part 1 and /2 is Part 2: what you need, every command and the agent prompt with a Copy button, the output that
means you are in, the live board and the usual fixes. /guide is hands-on/part2-evolve/APPLY-TO-YOUR-AGENT.md as a page.
The commands, the prompt and the page shell are the ones of the board's /join pages (join.py), so they cannot drift.

The pages link each other with relative links only and never name their own address, so the same files work under
any address. PUBLIC_BASE is the one place in the code that names it. When it changes, change with it what names the
address outside the pages: tools/public/README.md (tools/test_public_site.py fails until it agrees), the decks' stage
config and the prep email.
"""

from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path

from . import deploy, join, paths

GITHUB_REPO = "JoelNiklaus/aidev-harness-workshop"  # the public repo (runner launch.GITHUB_REPO)
PUBLIC_BASE = "https://niklaus.ai/aidev-harness-workshop"  # where GitHub Pages serves the public repo's docs/ today
BOARD = deploy.space_url(deploy.DEFAULT_SPACE)
REPO_URL = f"https://github.com/{GITHUB_REPO}"
GUIDE_SOURCE = f"{REPO_URL}/blob/main/hands-on/part2-evolve/APPLY-TO-YOUR-AGENT.md"
TITLES = {"1": "Part 1: join the swarm", "2": "Part 2: evolve the harness", "guide": "Guide: run the loop on your own agent"}
CLAUDE_INSTALL = "curl -fsSL https://claude.ai/install.sh | bash"
CODEX_INSTALL = "curl -fsSL https://chatgpt.com/codex/install.sh | sh"
COUPON = "the coupon link is on the slide and in the prep email"


def pages() -> dict[str, str]:
    """Path under docs/ -> file content."""
    swarm_url = join.wheel_url((paths.runner_dir() / "README.md").read_text())
    swarm = join.swarm_command(swarm_url)
    prompt = join.agent_prompt((paths.runner_dir() / "AGENT_PROMPT.md").read_text(), swarm_url)
    evolve = join.evolve_command(join.wheel_url((paths.evolve_dir() / "README.md").read_text()))
    setup = dict(uv_unix=join.box("uv-unix", join.UV_UNIX), uv_win=join.box("uv-win", join.UV_WINDOWS), login=join.box("login", join.LOGIN),
                 token_help=join.TOKEN_HELP, coupon=COUPON, board=BOARD)
    part1 = PART1.format(lead=join.LEAD, agents=join.AGENTS, check=join.box("check", f"{swarm} check"), prompt=join.box("prompt", prompt),
                         command=join.box("command", f"{swarm} launch"), fixes=_fixes(FIXES), nav=_nav("1"), **setup)
    part2 = PART2.format(lead=join.EVOLVE_LEAD, agent_switch=join.AGENT_SWITCH, blocks_label=join.BLOCKS_LABEL, show_help=join.SHOW_HELP,
                         claude_install=join.box("claude-install", CLAUDE_INSTALL, "for-claude"), codex_install=join.box("codex-install", CODEX_INSTALL, "for-codex"),
                         check=join.box("check", f"{evolve} check"), blocks=join.block_sections(evolve), show=join.box("show", f"{evolve} show"),
                         replay=join.box("replay", f"{evolve} replay"), fixes=_fixes(FIXES[:1] + AGENT_FIXES + FIXES[1:]), nav=_nav("2"), **setup)
    guide = GUIDE.format(nav=_nav("guide"), source=GUIDE_SOURCE, body=markdown((paths.evolve_dir() / "APPLY-TO-YOUR-AGENT.md").read_text()))
    index = INDEX.format(board=BOARD, repo=REPO_URL, **{f"title_{k}": v for k, v in TITLES.items()})
    return {"index.html": _page("Agent Harnesses workshop · AI Dev NYC", index),
            "1/index.html": _page(f"{TITLES['1']} · AI Dev NYC", part1),
            "2/index.html": _page(f"{TITLES['2']} · AI Dev NYC", part2, join.EVOLVE_STYLE, join.EVOLVE_SCRIPT),
            "guide/index.html": _page(f"{TITLES['guide']} · AI Dev NYC", guide),
            ".nojekyll": ""}


def write(out: Path) -> None:
    for rel, text in pages().items():
        p = Path(out) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _page(title: str, body: str, extra_style: str = "", extra_script: str = "") -> str:
    return join.html_page(title, body, SITE_STYLE + extra_style, extra_script).decode()


def _nav(current: str) -> str:
    """Links to the other pages, relative to a page one folder below the site's start page."""
    links = [("../", "Overview")] + [(f"../{name}/", title.split(":")[0]) for name, title in TITLES.items() if name != current]
    return '<nav class="nav">' + "".join(f'<a href="{href}">{text}</a>' for href, text in links) + "</nav>"


def _fixes(fixes: list[tuple[str, str]]) -> str:
    return "<ul class=\"fixes\">\n" + "\n".join(f"<li><b>{symptom}:</b> {fix}</li>" for symptom, fix in fixes) + "\n</ul>"


# --- markdown: the constructs the guide uses, nothing else ---

_CODE = re.compile(r"`([^`]+)`")
_LIST = re.compile(r"(- |\d+\. )(.*)")


def _inline(text: str) -> str:
    """Bold, code spans and https links; anything else that looks like markup is refused."""
    spans: list[str] = []

    def keep(m: re.Match) -> str:
        spans.append(f"<code>{escape(m.group(1), quote=False)}</code>")
        return f"\x00{len(spans) - 1}\x00"

    out = escape(_CODE.sub(keep, text), quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!!)\[([^\]]+)\]\((https://[^)\s]+)\)", r'<a href="\2">\1</a>', out)
    left = re.sub(r"<a href=\"[^\"]*\">", "", out)
    if any(mark in left for mark in ("**", "](", "![", "`")):
        raise ValueError(f"markdown the guide page does not render: {text[:80]!r}")
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], out)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def markdown(md: str) -> str:
    """Headings (# and ##), paragraphs, lists, tables and fenced code (each block in a Copy box) as HTML. A construct
    outside this set raises ValueError, so the export fails instead of publishing raw markup."""
    lines, out, i, n_code = md.splitlines(), [], 0, 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("```")), None)
            if end is None:
                raise ValueError("markdown: a code block is never closed")
            n_code += 1
            out.append(join.box(f"code-{n_code}", "\n".join(lines[i + 1:end])))
            i = end + 1
        elif line.startswith("#"):
            marks, _, title = line.partition(" ")
            if marks not in ("#", "##"):
                raise ValueError(f"markdown: heading level not rendered: {line[:80]!r}")
            out.append(f"<h{len(marks)}>{_inline(title)}</h{len(marks)}>")
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            if len(rows) < 3 or any(set(c) - set("-: ") for c in rows[1]) or any(len(r) != len(rows[0]) for r in rows):
                raise ValueError(f"markdown: not a table with a header, a separator and rows: {line[:80]!r}")
            head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f'<td data-label="{escape(re.sub(r"[*`]", "", h))}">{_inline(c)}</td>' for h, c in zip(rows[0], r)) + "</tr>\n"
                           for r in rows[2:])
            out.append(f"<table><thead><tr>{head}</tr></thead>\n<tbody>\n{body}</tbody></table>")
        elif _LIST.match(line):
            tag = "ul" if line.startswith("- ") else "ol"
            items: list[str] = []
            while i < len(lines) and lines[i].strip() and (_LIST.match(lines[i]) or lines[i].startswith("  ")):
                m = _LIST.match(lines[i])
                if m:
                    items.append(m.group(2))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(f"<{tag}>" + "\n".join(f"<li>{_inline(item)}</li>" for item in items) + f"</{tag}>")
        elif line.startswith((">", " ", "\t")):
            raise ValueError(f"markdown the guide page does not render: {line[:80]!r}")
        else:
            para, i = [line.strip()], i + 1  # always consumes its first line, so the loop cannot stall
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```", ">")) and not _LIST.match(lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out) + "\n"


# --- page texts ---

# Symptom -> fix, as in the prep email. Part 2 adds the coding agent's own login after the first entry.
FIXES = [
    ("<code>uvx: command not found</code>", "open a new terminal window after installing, so it picks up the new command."),
    ("&quot;no credits&quot; or 402", f"open the coupon link while logged in to Hugging Face; {COUPON}."),
    ("401 or 403", "the token lacks a permission. Create a new one with both permissions from step 2 and run <code>uvx hf auth login</code> again."),
    ("&quot;No Hugging Face token found&quot;", "run <code>uvx hf auth login</code> and paste your token."),
    ("Timeouts or SSL errors", "a company VPN or proxy may block the calls. Turn the VPN off or use another network, for example your phone's hotspot."),
]
AGENT_FIXES = [
    ("<code>claude</code> or <code>codex</code>: <code>command not found</code>", "open a new terminal window after installing the coding agent."),
    ("&quot;Not logged in&quot; from <code>claude</code>, or 401 from <code>codex</code>", "run <code>claude</code> and type <code>/login</code>, or run <code>codex login</code>."),
]

SITE_STYLE = """html,body{font-size:18px;overflow-wrap:anywhere}
h1{font-size:30px}
h2{font-size:22px}
h2 b{width:36px;height:36px}
.label{font-size:17px}
pre{font-size:15px}
pre.out{padding-bottom:14px;border:1px solid var(--line);border-radius:12px;background:var(--code);color:var(--mute);margin:6px 0}
code{font:500 .88em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);border-radius:6px;padding:1px 5px}
ul,ol{margin:8px 0;padding-left:24px}
li{margin:8px 0}
.nav{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.nav a{display:inline-flex;align-items:center;min-height:48px;padding:6px 16px;border:2px solid var(--line);border-radius:12px;font-weight:700;text-decoration:none}
.cards a{display:block;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin:14px 0;text-decoration:none;color:var(--fg)}
.cards a b{display:block;color:var(--acc);font-size:22px}
.doc h2{display:block;margin:28px 0 8px}
table{border-collapse:collapse;width:100%;margin:12px 0}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
@media (max-width:699px){thead{display:none} table,tbody,tr,td{display:block} tr{margin:12px 0} td{border-width:0 1px 1px} tr td:first-child{border-top-width:1px;background:var(--panel)}
td:not(:first-child)::before{content:attr(data-label);display:block;color:var(--mute);font-size:15px}}
@media (min-width:700px){html,body{font-size:20px} h1{font-size:40px} h2{font-size:26px} .label{font-size:18px} pre{font-size:17px} button{font-size:18px}}
"""

_SETUP_STEPS = """<section>
<h2><b>1</b>Install uv</h2>
<p class="label">macOS and Linux</p>
{uv_unix}
<p class="label">Windows (PowerShell)</p>
{uv_win}
</section>

<section>
<h2><b>2</b>Get a Hugging Face token</h2>
<p>{token_help}</p>
{login}
</section>
"""

PART1 = """<h1>Part 1: join the swarm</h1>
<p class="lead">{lead}</p>
<a class="board" href="{board}/">Live board</a>

<section>
<h2>What you need</h2>
<ul>
<li>A <b>Hugging Face account</b> with credits. Sign up for free at <a href="https://huggingface.co/join">huggingface.co/join</a>. The session coupon adds $10 of credits; {coupon}. Open it while logged in.</li>
<li><b>uv</b> (step 1) and a Hugging Face <b>token</b> (step 2).</li>
<li>A laptop with a terminal. No Docker, no GPU, no repo to clone.</li>
</ul>
</section>

""" + _SETUP_STEPS + """
<section>
<h2><b>3</b>Check your setup</h2>
<p>This makes one GLM-5.3-Flash call, runs one tiny Hugging Face Job and one command in a Hugging Face Sandbox. It takes 15 to 35 seconds, costs well under one cent and claims no workshop task.</p>
{check}
<p class="label">The last line reads</p>
<pre class="out">[aidev-swarm HH:MM:SS] ready: &lt;you&gt; can call the workshop model, run HF Jobs and use HF Sandboxes. See you at the workshop.</pre>
</section>

<section>
<h2><b>4</b>Start your Job</h2>
<p>Paste this into your coding agent. {agents}</p>
{prompt}
<p class="label">Or run it yourself in a terminal</p>
{command}
</section>

<section>
<h2>You are in when you see this</h2>
<pre class="out">[aidev-swarm HH:MM:SS] launched HF Job &lt;job id&gt; in &lt;you&gt;'s namespace: it runs trials until the grid is full, up to 8 at once on Hugging Face. You can close the laptop.
  Job page: https://huggingface.co/jobs/&lt;you&gt;/&lt;job id&gt;
  Logs:     uvx hf jobs logs -f &lt;job id&gt;
  Stop:     uvx hf jobs cancel &lt;job id&gt;
  Board:    {board}</pre>
<p>The command returns in a few seconds and the Job keeps running on Hugging Face. Each finished trial appears on the board.</p>
<a class="board" href="{board}/">Open the live board</a>
</section>

<section>
<h2>If something fails</h2>
{fixes}
</section>

{nav}
"""

PART2 = """<h1>Part 2: evolve the harness</h1>
<p class="lead">{lead}</p>
<a class="board" href="{board}/evolve">Live board</a>
{agent_switch}

<section>
<h2>What you need</h2>
<ul>
<li>A <b>Hugging Face account</b> with credits. The session coupon adds $10 of credits, which covers one run; {coupon}. Open it while logged in.</li>
<li><b>uv</b> (step 1) and a Hugging Face <b>token</b> (step 2), as in Part 1.</li>
<li>A <b>paid Claude plan</b> that includes Claude Code, or a <b>paid ChatGPT plan</b> that includes Codex. Your coding agent runs on your own plan (step 3).</li>
<li>A laptop that stays open during the run.</li>
</ul>
</section>

""" + _SETUP_STEPS + """
<section>
<h2><b>3</b>Install your coding agent</h2>
<p>Skip this step if Claude Code or Codex already runs on your laptop and is logged in.</p>
<div class="for-claude"><p class="label">Claude Code, macOS and Linux. Then run <code>claude</code> and log in.</p></div>
{claude_install}
<div class="for-codex"><p class="label">Codex, macOS and Linux. Then run <code>codex login</code>.</p></div>
{codex_install}
<p>On Windows, use the PowerShell installer for <a href="https://code.claude.com/docs/en/setup">Claude Code</a> or <a href="https://developers.openai.com/codex/cli">Codex</a>.</p>
</section>

<section>
<h2><b>4</b>Check your setup</h2>
<p>Hugging Face login, your coding agent and the board, in a few seconds:</p>
{check}
<p class="label">The last line reads (or <code>codex</code>)</p>
<pre class="out">ready: start your loop with `aidev-evolve run --proposer claude`</pre>
</section>

<p class="label">5 &nbsp; {blocks_label}</p>
{blocks}<section>
<h2>You are in when you see this</h2>
<pre class="out">Strategy &lt;the strategy of your block&gt;
Workshop board: {board}/evolve  (your progress appears there; --no-board keeps it private)</pre>
<p>The terminal then shows the proposed patch with its hypothesis and diff, validation before and after, KEPT or DROPPED, and the held-out check. Your run also appears on the board.</p>
<a class="board" href="{board}/evolve">Open the live board</a>
</section>

<section>
<h2><b>6</b>Afterwards: open your final agent</h2>
<p>{show_help}</p>
{show}
<p class="label">No Wi-Fi? Play a recorded run, offline</p>
{replay}
<p>To run the loop on your own agent after the workshop, read the <a href="../guide/">guide</a>.</p>
</section>

<section>
<h2>If something fails</h2>
{fixes}
</section>

{nav}
"""

GUIDE = """{nav}
<article class="doc">
{body}</article>
<p class="label">This page is <a href="{source}">APPLY-TO-YOUR-AGENT.md</a> of the public repo.</p>
{nav}
"""

INDEX = """<h1>Agent Harnesses workshop, AI Dev NYC 2026</h1>
<p class="lead">Everything you need for the two hands-on sessions: what to install, the commands to copy, and what to do when a step fails.</p>
<div class="cards">
<a href="1/"><b>{title_1}</b>Mon 30 Nov. Your laptop starts one Hugging Face Job that runs Terminal-Bench 2.1 tasks in different agent harnesses, and we fill one live leaderboard together.</a>
<a href="2/"><b>{title_2}</b>Tue 1 Dec. Your coding agent proposes changes to a small agent's harness, and a loop keeps only the changes that score better.</a>
<a href="guide/"><b>{title_guide}</b>After the workshop. What you need to bring, and a toy example that runs in 3 minutes for under one cent.</a>
</div>
<nav class="nav"><a href="{board}/">Live board</a><a href="{repo}">Code on GitHub</a></nav>
"""


if __name__ == "__main__":
    write(Path(sys.argv[1]))
    print(f"attendee site written to {sys.argv[1]}: " + ", ".join(f"{PUBLIC_BASE}/{n}" for n in TITLES))
