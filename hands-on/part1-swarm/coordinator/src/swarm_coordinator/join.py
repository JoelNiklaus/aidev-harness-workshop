"""GET /join: what an attendee opens from the slide's QR code to start (install uv, token, launch).

Large type for phones and laptops; every command sits in a box with a Copy button. Nothing here is typed by
hand: the wheel URL is the line `<WHEEL_URL>` is today `...` of runner/README.md (a runner test pins it to
`launch.default_package_url()`, and the deck's numbers_extract.py reads it through `wheel_url` below), and
the prompt is the quoted paragraph of runner/AGENT_PROMPT.md with that URL filled in.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

PATH = "/join"
WHEEL_LINE = re.compile(r"`<WHEEL_URL>` is today `([^`]+)`")


def wheel_url(readme: str) -> str:
    m = WHEEL_LINE.search(readme)
    if not m:
        raise ValueError("runner README lacks the line: `<WHEEL_URL>` is today `<url>`")
    return m.group(1)


def launch_command(url: str) -> str:
    return f'uvx --from "{url}" aidev-swarm launch'


def agent_prompt(prompt_md: str, url: str) -> str:
    """The quoted paragraph of AGENT_PROMPT.md, with the wheel URL filled in."""
    quoted = [ln.removeprefix("> ") for ln in prompt_md.splitlines() if ln.startswith("> ")]
    prompt = " ".join(quoted).replace("<WHEEL_URL>", url)
    if launch_command(url) not in prompt:
        raise ValueError("AGENT_PROMPT.md's quoted prompt must contain `uvx --from \"<WHEEL_URL>\" aidev-swarm launch`")
    return prompt


def _box(box_id: str, text: str) -> str:
    return (f'<div class="box"><pre id="{box_id}">{escape(text, quote=False)}</pre>'
            f'<button type="button" data-copy="{box_id}">Copy</button></div>')


def page(runner_dir: Path) -> bytes:
    runner_dir = Path(runner_dir)
    url = wheel_url((runner_dir / "README.md").read_text())
    prompt = agent_prompt((runner_dir / "AGENT_PROMPT.md").read_text(), url)
    return TEMPLATE.format(
        uv_unix=_box("uv-unix", "curl -LsSf https://astral.sh/uv/install.sh | sh"),
        uv_win=_box("uv-win", 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'),
        login=_box("login", "uvx hf auth login"),
        prompt=_box("prompt", prompt),
        command=_box("command", launch_command(url)),
    ).encode()


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Join the swarm · AI Dev NYC</title>
<style>
:root{{--bg:#0b0f17;--panel:#121826;--line:#243047;--fg:#f3f6fb;--mute:#a3b0c4;--acc:#ffd21e;--code:#05080e}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:var(--bg);color:var(--fg);font:500 22px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:20px 16px 48px}}
h1{{font-size:36px;line-height:1.15;margin:8px 0 6px}}
.lead{{color:var(--mute);margin:0 0 18px}}
a{{color:var(--acc)}}
.board{{display:inline-block;font-weight:700;text-decoration:none;border:2px solid var(--acc);border-radius:12px;padding:8px 16px;margin:4px 0 8px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin:18px 0}}
h2{{font-size:28px;margin:0 0 8px;display:flex;gap:12px;align-items:center}}
h2 b{{flex:none;display:inline-grid;place-items:center;width:42px;height:42px;border-radius:50%;background:var(--acc);color:#111}}
p{{margin:8px 0}}
.label{{color:var(--mute);font-size:20px;margin:12px 0 4px}}
.box{{position:relative;background:var(--code);border:1px solid var(--line);border-radius:12px;margin:6px 0}}
pre{{margin:0;padding:14px 16px 58px;font:500 19px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}}
button{{position:absolute;right:10px;bottom:10px;font:700 20px system-ui,sans-serif;padding:8px 22px;border:0;border-radius:10px;background:var(--acc);color:#111;cursor:pointer;min-height:44px}}
button.ok{{background:#22c55e}}
@media (min-width:700px){{html,body{{font-size:26px}} h1{{font-size:48px}} h2{{font-size:32px}} .label{{font-size:22px}} pre{{font-size:22px}} button{{font-size:22px}}}}
</style>
</head>
<body>
<main>
<h1>Join the swarm</h1>
<p class="lead">Your coding agent starts one job on Hugging Face that runs trials for the room's board. Your laptop can close afterwards.</p>
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
<p>At <a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a>, create a token with the permissions <b>Make calls to Inference Providers</b> and <b>Start and manage Jobs</b> (a <b>write</b> token also works). Then log in and paste it:</p>
{login}
</section>

<section>
<h2><b>3</b>Paste this into your coding agent</h2>
<p>Claude Code, Codex, Cursor or any agent that can run shell commands:</p>
{prompt}
<p class="label">Or run it yourself in a terminal</p>
{command}
</section>

<p><a href="/">See your results on the live board</a></p>
</main>
<script>
for (const b of document.querySelectorAll("button[data-copy]")) {{
  b.addEventListener("click", async () => {{
    const el = document.getElementById(b.dataset.copy);
    try {{ await navigator.clipboard.writeText(el.textContent); }}
    catch {{  // no clipboard API (plain http, old browser): select the text and copy it
      const r = document.createRange(); r.selectNodeContents(el);
      const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand("copy");
    }}
    b.textContent = "Copied"; b.classList.add("ok");
    setTimeout(() => {{ b.textContent = "Copy"; b.classList.remove("ok"); }}, 1500);
  }});
}}
</script>
</body>
</html>
"""
