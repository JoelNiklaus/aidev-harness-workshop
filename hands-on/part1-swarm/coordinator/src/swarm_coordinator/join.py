"""GET /join and GET /evolve/join: the board's own start pages for Part 1 (install uv, token, launch) and Part 2 (the
run command of each seating block). The attendee site on GitHub Pages (attendee_site.py) builds its pages from the same
commands, prompt and page shell, so the two cannot show different commands.

Large type for phones and laptops; every command sits in a box with a Copy button. Nothing here is typed by
hand: the wheel URL is the line `<WHEEL_URL>` is today `...` of runner/README.md (a runner test pins it to
`launch.default_package_url()`, and the deck's numbers_extract.py reads it through `wheel_url` below), and
the prompt is the quoted paragraph of runner/AGENT_PROMPT.md with that URL filled in. The Part 2 page takes its
wheel URL from the same line of the Part 2 README and its blocks from BLOCKS (a test compares it with the package's).
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

PATH = "/join"
EVOLVE_PATH = "/evolve/join"
# Seating block -> strategy preset: a copy of aidev_evolve.strategy.BLOCKS (tests/test_evolve_join.py compares them).
BLOCKS = {"left": "baseline", "middle": "every-weakness", "right": "code-audit"}
BLOCK_HINT = {"baseline": "the proposer reads the trajectories of the search tasks and fixes what it sees",
              "every-weakness": "the proposer tallies every failure it finds in the trajectories and fixes each one seen twice",
              "code-audit": "the proposer reads agent.py line by line, then the trajectories"}
WHEEL_LINE = re.compile(r"`<WHEEL_URL>` is today `([^`]+)`")
UV_UNIX = "curl -LsSf https://astral.sh/uv/install.sh | sh"
UV_WINDOWS = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
LOGIN = "uvx hf auth login"


def wheel_url(readme: str) -> str:
    m = WHEEL_LINE.search(readme)
    if not m:
        raise ValueError("runner README lacks the line: `<WHEEL_URL>` is today `<url>`")
    return m.group(1)


def swarm_command(url: str) -> str:
    return f'uvx --from "{url}" aidev-swarm'


def launch_command(url: str) -> str:
    return f"{swarm_command(url)} launch"


def evolve_command(url: str) -> str:
    return f'uvx --from "{url}" aidev-evolve'


def agent_prompt(prompt_md: str, url: str) -> str:
    """The quoted paragraph of AGENT_PROMPT.md, with the wheel URL filled in."""
    quoted = [ln.removeprefix("> ") for ln in prompt_md.splitlines() if ln.startswith("> ")]
    prompt = " ".join(quoted).replace("<WHEEL_URL>", url)
    if launch_command(url) not in prompt:
        raise ValueError("AGENT_PROMPT.md's quoted prompt must contain `uvx --from \"<WHEEL_URL>\" aidev-swarm launch`")
    return prompt


def box(box_id: str, text: str, cls: str = "") -> str:
    return (f'<div class="box{" " + cls if cls else ""}"><pre id="{box_id}">{escape(text, quote=False)}</pre>'
            f'<button type="button" data-copy="{box_id}">Copy</button></div>')


def html_page(title: str, body: str, extra_style: str = "", extra_script: str = "") -> bytes:
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{title}</title>\n'
            f"<style>\n{STYLE}{extra_style}</style>\n</head>\n<body>\n<main>\n{body}</main>\n"
            f"<script>\n{COPY_SCRIPT}{extra_script}</script>\n</body>\n</html>\n").encode()


def page(runner_dir: Path) -> bytes:
    runner_dir = Path(runner_dir)
    url = wheel_url((runner_dir / "README.md").read_text())
    prompt = agent_prompt((runner_dir / "AGENT_PROMPT.md").read_text(), url)
    return html_page("Join the swarm · AI Dev NYC", BODY.format(
        lead=LEAD, token_help=TOKEN_HELP, agents=AGENTS,
        uv_unix=box("uv-unix", UV_UNIX),
        uv_win=box("uv-win", UV_WINDOWS),
        login=box("login", LOGIN),
        prompt=box("prompt", prompt),
        command=box("command", launch_command(url)),
    ))


def evolve_page(readme: Path) -> bytes:
    """The Part 2 start page: check, the run command of each seating block (Claude Code or Codex), the final agent,
    the offline replay."""
    cmd = evolve_command(wheel_url(Path(readme).read_text()))
    return html_page("Evolve the harness · AI Dev NYC", EVOLVE_BODY.format(
        lead=EVOLVE_LEAD, agent_switch=AGENT_SWITCH, blocks_label=BLOCKS_LABEL, show_help=SHOW_HELP,
        check=box("check", f"{cmd} check"), blocks=block_sections(cmd), show=box("show", f"{cmd} show"), replay=box("replay", f"{cmd} replay")),
        EVOLVE_STYLE, EVOLVE_SCRIPT)


def block_sections(cmd: str) -> str:
    """One section per seating block with its run command, for Claude Code and for Codex (AGENT_SWITCH picks which shows)."""
    blocks = ""
    for i, (block, preset) in enumerate(BLOCKS.items()):
        blocks += (f'<section id="block-{block}">\n<h2><b>{"ABC"[i]}</b>{block.capitalize()} block: {preset}</h2>\n'
                   f"<p>{escape(BLOCK_HINT[preset])}.</p>\n"
                   + box(f"run-{block}", f"{cmd} run --block {block}", "for-claude") + "\n"
                   + box(f"run-{block}-codex", f"{cmd} run --block {block} --proposer codex", "for-codex") + "\n</section>\n\n")
    return blocks


STYLE = """:root{--bg:#0b0f17;--panel:#121826;--line:#243047;--fg:#f3f6fb;--mute:#a3b0c4;--acc:#ffd21e;--code:#05080e}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font:500 22px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:36px;line-height:1.15;margin:8px 0 6px}
.lead{color:var(--mute);margin:0 0 18px}
a{color:var(--acc)}
.board{display:inline-block;font-weight:700;text-decoration:none;border:2px solid var(--acc);border-radius:12px;padding:8px 16px;margin:4px 0 8px}
section{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin:18px 0}
h2{font-size:28px;margin:0 0 8px;display:flex;gap:12px;align-items:center}
h2 b{flex:none;display:inline-grid;place-items:center;width:42px;height:42px;border-radius:50%;background:var(--acc);color:#111}
p{margin:8px 0}
.label{color:var(--mute);font-size:20px;margin:12px 0 4px}
.box{position:relative;background:var(--code);border:1px solid var(--line);border-radius:12px;margin:6px 0}
pre{margin:0;padding:14px 16px 58px;font:500 19px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
button{position:absolute;right:10px;bottom:10px;font:700 20px system-ui,sans-serif;padding:8px 22px;border:0;border-radius:10px;background:var(--acc);color:#111;cursor:pointer;min-height:44px}
button.ok{background:#22c55e}
@media (min-width:700px){html,body{font-size:26px} h1{font-size:48px} h2{font-size:32px} .label{font-size:22px} pre{font-size:22px} button{font-size:22px}}
"""

COPY_SCRIPT = """for (const b of document.querySelectorAll("button[data-copy]")) {
  b.addEventListener("click", async () => {
    const el = document.getElementById(b.dataset.copy);
    try { await navigator.clipboard.writeText(el.textContent); }
    catch {  // no clipboard API (plain http, old browser): select the text and copy it
      const r = document.createRange(); r.selectNodeContents(el);
      const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand("copy");
    }
    b.textContent = "Copied"; b.classList.add("ok");
    setTimeout(() => { b.textContent = "Copy"; b.classList.remove("ok"); }, 1500);
  });
}
"""

LEAD = "Your coding agent starts one job on Hugging Face that runs trials for the shared board on the projector. Your laptop can close afterwards."
TOKEN_HELP = ('At <a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a>, create a token with the permissions '
              "<b>Make calls to Inference Providers</b> and <b>Start and manage Jobs</b> (a <b>write</b> token also works). Then log in and paste it:")
AGENTS = "Claude Code, Codex, Cursor or any agent that can run shell commands:"

BODY = """<h1>Join the swarm</h1>
<p class="lead">{lead}</p>
<a class="board" href="/">&#8592; Live board</a>

<section>
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

<section>
<h2><b>3</b>Paste this into your coding agent</h2>
<p>{agents}</p>
{prompt}
<p class="label">Or run it yourself in a terminal</p>
{command}
</section>

<p><a href="/">See your results on the live board</a></p>
"""

EVOLVE_STYLE = """.agent{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0 0}
.agent button{position:static;background:var(--panel);color:var(--fg);border:2px solid var(--line)}
body:not(.codex) .agent button[data-agent="claude"],body.codex .agent button[data-agent="codex"]{background:var(--acc);color:#111;border-color:var(--acc)}
body.codex .for-claude{display:none}
body:not(.codex) .for-codex{display:none}
"""

EVOLVE_SCRIPT = """for (const b of document.querySelectorAll("button[data-agent]")) {
  b.addEventListener("click", () => document.body.classList.toggle("codex", b.dataset.agent === "codex"));
}
"""

EVOLVE_LEAD = ("Your coding agent proposes patches to a small agent; a job on Hugging Face scores each one, and the loop keeps a patch only "
               "if it wins on tasks your coding agent never saw. One run takes about 20 minutes.")
AGENT_SWITCH = ('<div class="agent"><span>My coding agent:</span><button type="button" data-agent="claude">Claude Code</button>'
                '<button type="button" data-agent="codex">Codex</button></div>')
BLOCKS_LABEL = "Start the run of your seating block (as seen from your seat, facing the stage)"
SHOW_HELP = "The run ends with the path of <code>final_agent.py</code>. This prints it, with the seed weaknesses that are still in it:"

EVOLVE_BODY = """<h1>Evolve the harness</h1>
<p class="lead">{lead}</p>
<a class="board" href="/evolve">&#8592; Live board</a>
{agent_switch}

<section>
<h2><b>1</b>Check your setup</h2>
<p>Hugging Face login, your coding agent and the board, in a few seconds:</p>
{check}
</section>

<p class="label">2 &nbsp; {blocks_label}</p>
{blocks}<section>
<h2><b>3</b>Afterwards: open your final agent</h2>
<p>{show_help}</p>
{show}
<p class="label">No Wi-Fi? Play a recorded run, offline</p>
{replay}
</section>

<p><a href="/evolve">See your run on the live board</a></p>
"""
