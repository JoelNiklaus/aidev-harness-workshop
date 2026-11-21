"""Short links for the slides: static redirect pages for GitHub Pages (the public repo's docs/ folder).

    python3 -m swarm_coordinator.shortlinks <public checkout>/docs      # tools/export_public.sh runs this

Joel's GitHub user site has the custom domain niklaus.ai, so the project site of the public repo is served at
https://niklaus.ai/aidev-harness-workshop/ (https://joelniklaus.github.io/aidev-harness-workshop/ redirects there).
/1 opens the Part 1 join page of the workshop board, /2 the Part 2 start page, /guide the apply-to-your-agent guide.
Nothing to maintain: the targets come from the deploy target and the join paths, and a re-export rewrites the pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import deploy, join

GITHUB_REPO = "JoelNiklaus/aidev-harness-workshop"  # the public repo (runner launch.GITHUB_REPO)
PUBLIC_BASE = "https://niklaus.ai/aidev-harness-workshop"
GUIDE = f"https://github.com/{GITHUB_REPO}/blob/main/hands-on/part2-evolve/APPLY-TO-YOUR-AGENT.md"


def _links() -> dict[str, tuple[str, str]]:
    board = deploy.space_url(deploy.DEFAULT_SPACE)
    return {"1": (board + join.PATH, "Part 1: join the swarm"), "2": (board + join.EVOLVE_PATH, "Part 2: evolve the harness"),
            "guide": (GUIDE, "Run the loop on your own agent")}


LINKS = _links()

REDIRECT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={url}">
<link rel="canonical" href="{url}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>location.replace("{url}");</script>
</head><body style="font:22px system-ui,sans-serif;padding:24px">
<p>{title}: <a href="{url}">{url}</a></p>
</body></html>
"""

INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Harnesses workshop · AI Dev NYC</title></head>
<body style="font:22px/1.5 system-ui,sans-serif;padding:24px;max-width:900px;margin:0 auto">
<h1>Agent Harnesses workshop, AI Dev NYC 2026</h1>
<ul>
{items}
<li><a href="https://github.com/{repo}">Code</a></li>
</ul>
</body></html>
"""


def pages() -> dict[str, str]:
    out = {f"{name}/index.html": REDIRECT.format(url=url, title=title) for name, (url, title) in LINKS.items()}
    items = "\n".join(f'<li>{title}: <a href="{PUBLIC_BASE}/{name}">{PUBLIC_BASE}/{name}</a></li>' for name, (_, title) in LINKS.items())
    out["index.html"] = INDEX.format(items=items, repo=GITHUB_REPO)
    out[".nojekyll"] = ""
    return out


def write(out: Path) -> None:
    for rel, text in pages().items():
        p = Path(out) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
    print(f"short links written to {sys.argv[1]}: " + ", ".join(f"{PUBLIC_BASE}/{n}" for n in LINKS))
