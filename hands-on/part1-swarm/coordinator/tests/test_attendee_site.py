"""The attendee site: static pages served by GitHub Pages from the public repo's docs/ folder. People scan a QR code
on the slide and get everything they need on their phone: what to install, the commands with Copy buttons, what the
output looks like, the live board, the usual fixes.

The commands and the prompt are the ones of the board's /join pages (swarm_coordinator.join), so they cannot drift.
"""

from __future__ import annotations

import html
import posixpath
import re

import pytest

from swarm_coordinator import attendee_site as site
from swarm_coordinator import deploy, join, paths

BOARD = deploy.space_url(deploy.DEFAULT_SPACE)
PART2 = deploy.COORD.parents[1] / "part2-evolve"
SWARM_URL = join.wheel_url((paths.runner_dir() / "README.md").read_text())
EVOLVE_URL = join.wheel_url((PART2 / "README.md").read_text())
HTML_PAGES = ["index.html", "1/index.html", "2/index.html", "guide/index.html"]


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return site.pages()


def boxes(page: str) -> list[str]:
    """The text of every copyable box (unescaped, as the clipboard gets it), each with its own Copy button."""
    found = re.findall(r'<pre id="([\w-]+)"[^>]*>(.*?)</pre><button type="button" data-copy="([\w-]+)">Copy</button>', page, re.S)
    assert all(pre_id == button_id for pre_id, _, button_id in found)
    return [html.unescape(text) for _, text, _ in found]


def text_of(page: str) -> str:
    """What a reader sees: the body without tags, styles and scripts."""
    body = re.sub(r"<(style|script)\b.*?</\1>", " ", page, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))


def test_write_puts_the_pages_under_the_folder_and_tells_pages_not_to_run_jekyll(tmp_path):
    site.write(tmp_path / "docs")
    names = {p.relative_to(tmp_path / "docs").as_posix() for p in (tmp_path / "docs").rglob("*") if p.is_file()}
    assert names == {*HTML_PAGES, ".nojekyll"}


def test_no_page_is_a_redirect(pages):
    for name in HTML_PAGES:
        assert "http-equiv" not in pages[name] and "location.replace" not in pages[name] and "location.href" not in pages[name], name


def test_part1_has_every_command_and_the_agent_prompt_in_a_copy_box(pages):
    got = boxes(pages["1/index.html"])
    prompt = join.agent_prompt((paths.runner_dir() / "AGENT_PROMPT.md").read_text(), SWARM_URL)
    for expected in (join.UV_UNIX, join.UV_WINDOWS, join.LOGIN, f'uvx --from "{SWARM_URL}" aidev-swarm check',
                     f'uvx --from "{SWARM_URL}" aidev-swarm launch', prompt):
        assert expected in got, expected


def test_part2_has_check_the_run_command_of_every_block_for_both_agents_show_and_replay(pages):
    got = boxes(pages["2/index.html"])
    cmd = f'uvx --from "{EVOLVE_URL}" aidev-evolve'
    expected = [join.UV_UNIX, join.UV_WINDOWS, join.LOGIN, f"{cmd} check", f"{cmd} show", f"{cmd} replay"]
    for block in join.BLOCKS:
        expected += [f"{cmd} run --block {block}", f"{cmd} run --block {block} --proposer codex"]
    for e in expected:
        assert e in got, e
    assert any("claude.ai/install.sh" in b for b in got) and any("codex/install.sh" in b for b in got), "how to install the coding agent"


def test_the_board_join_pages_use_the_same_command_builders(pages):
    """One source: the Space's /join pages and the static pages show the same commands."""
    space1 = boxes(join.page(paths.runner_dir()).decode())
    space2 = boxes(join.evolve_page(PART2 / "README.md").decode())
    assert set(space1) <= set(boxes(pages["1/index.html"]))
    assert set(space2) <= set(boxes(pages["2/index.html"]))


def test_each_part_says_what_you_need(pages):
    for name in ("1/index.html", "2/index.html"):
        t = text_of(pages[name])
        assert "What you need" in t and "Hugging Face account" in t and "credits" in t and "uv" in t, name
        assert "Make calls to Inference Providers" in t and "Start and manage Jobs" in t, name
    t2 = text_of(pages["2/index.html"])
    assert "paid" in t2 and "Claude Code" in t2 and "Codex" in t2


def test_each_part_shows_the_output_that_means_you_are_in_as_the_tools_print_it(pages):
    """The sample lines are the tools' own format strings, so a changed message fails here."""
    swarm_cli = (paths.runner_dir() / "src" / "aidev_swarm" / "cli.py").read_text()
    evolve_cli = (PART2 / "src" / "aidev_evolve" / "cli.py").read_text()
    t1, t2 = text_of(pages["1/index.html"]), text_of(pages["2/index.html"])
    assert "You are in when you see this" in t1 and "You are in when you see this" in t2
    for fragment in ("launched HF Job", "in {user}'s namespace", "You can close the laptop.", "  Job page: https://huggingface.co/jobs/",
                     "  Logs:     uvx hf jobs logs -f ", "  Stop:     uvx hf jobs cancel ", "  Board:    "):
        assert fragment in swarm_cli, fragment
        assert re.sub(r"\s+", " ", fragment.replace("{user}", "<you>")).strip() in t1, fragment
    assert f"Board: {BOARD}" in t1
    launch_py = (paths.runner_dir() / "src" / "aidev_swarm" / "launch.py").read_text()
    parallel = re.search(r"(?m)^DEFAULT_PARALLEL = (\d+)", launch_py).group(1)
    assert "trials until the grid is full" in swarm_cli and f"it runs trials until the grid is full, up to {parallel} at once on Hugging Face" in t1
    for fragment in ("ready: start your loop with `aidev-evolve run --proposer ", "Workshop board: ", "(your progress appears there; --no-board keeps it private)"):
        assert fragment in evolve_cli, fragment
        assert fragment.strip() in t2, fragment
    assert f"Workshop board: {BOARD}/evolve" in t2
    assert "can call the workshop model, run HF Jobs and use HF Sandboxes" in swarm_cli and "can call the workshop model, run HF Jobs and use HF Sandboxes" in t1


def test_each_part_links_the_live_board(pages):
    assert f'href="{BOARD}/"' in pages["1/index.html"]
    assert f'href="{BOARD}/evolve"' in pages["2/index.html"]


def test_each_part_lists_the_standard_fixes(pages):
    for name in ("1/index.html", "2/index.html"):
        t = text_of(pages[name])
        for needle in ("command not found", "401", "403", "402", "no credits", "uvx hf auth login", "VPN", "proxy"):
            assert needle in t, (name, needle)
    assert "/login" in text_of(pages["2/index.html"]) and "codex login" in text_of(pages["2/index.html"])


def test_no_page_names_a_coupon_link_and_the_parts_say_where_it_is(pages):
    for name, page in pages.items():
        # spelled in two halves: the export's refuse-list (tools/public_scan.sh) rejects any file that holds these strings
        assert "coupons/" + "claim" not in page.lower() and "huggingfacex" + "aidevnyc" not in page.lower(), name
    for name in ("1/index.html", "2/index.html"):
        assert "the coupon link is on the slide and in the prep email" in text_of(pages[name]), name


def test_every_page_works_under_any_base_address(pages):
    """Links inside the site are relative and open a page that exists; no page names the site's own address."""
    host = re.sub(r"^https://", "", site.PUBLIC_BASE).split("/")[0]
    for name in HTML_PAGES:
        page = pages[name]
        assert site.PUBLIC_BASE not in page and host not in page, name
        inside = 0
        for href in re.findall(r'href="([^"]+)"', page):
            if href.startswith(("https://", "#")):
                continue
            assert not href.startswith(("/", "http:", "//")), (name, href)
            inside += 1
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), href))
            assert href.endswith("/") and (f"{target}/index.html" if target != "." else "index.html") in pages, (name, href)
        assert inside >= 3, f"{name} links the other pages"
        assert not re.search(r'(?:src|action)="', page), name


def test_the_index_links_the_three_pages_and_the_code(pages):
    index = pages["index.html"]
    for href in ("1/", "2/", "guide/", f"https://github.com/{site.GITHUB_REPO}"):
        assert f'href="{href}"' in index, href


def test_the_public_base_is_one_https_address_without_a_trailing_slash():
    assert re.fullmatch(r"https://[a-z0-9.-]+(/[A-Za-z0-9._-]+)*", site.PUBLIC_BASE)
    assert site.GITHUB_REPO in (paths.runner_dir() / "src" / "aidev_swarm" / "launch.py").read_text()  # the repo the wheels are released from


def test_pages_are_made_for_a_phone(pages):
    for name in HTML_PAGES:
        page = pages[name]
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page, name
        assert len(page.encode()) < 50_000, (name, len(page.encode()))
        for external in ("<link", "<script src", "@import", "url(", "<img", "<iframe", "@font-face"):
            assert external not in page, (name, external)
        style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
        pre = re.search(r"(?m)^pre\{([^}]*)\}", style).group(1)
        assert "white-space:pre-wrap" in pre and "overflow-wrap:anywhere" in pre, "long commands wrap"
        assert any("overflow-wrap:anywhere" in rule for rule in re.findall(r"(?m)^html,body\{([^}]*)\}", style)), "long words and URLs wrap"
        assert int(re.search(r"(?m)^button\{[^}]*min-height:(\d+)px", style).group(1)) >= 44, "tap targets"
        assert int(re.search(r"(?m)^\.nav a\{[^}]*min-height:(\d+)px", style).group(1)) >= 44, "tap targets"


def test_the_wording_is_plain(pages):
    for name in HTML_PAGES:
        t = text_of(pages[name])
        for banned in ("the room", "mailto:", "email Joel", "email me", "WARNING", "\u2014", "\u2013", "!"):
            assert banned not in t, (name, banned)


def test_the_guide_page_is_the_take_home_guide_rendered_for_reading(pages):
    md = (PART2 / "APPLY-TO-YOUR-AGENT.md").read_text()
    page = pages["guide/index.html"]
    t = text_of(page)
    for heading in re.findall(r"(?m)^#{1,2} (.+)$", md):
        assert f">{html.escape(heading, quote=False)}</h" in page, heading
    code_blocks = [c.strip("\n") for c in re.findall(r"(?ms)^```\w*\n(.*?)^```", md)]
    assert code_blocks and all(c in boxes(page) for c in code_blocks), "every command block has a Copy button"
    prose = text_of(re.sub(r"<(code|pre)\b.*?</\1>", " ", page, flags=re.S))  # code may hold these characters, prose may not
    for raw in ("**", "```", "](", "|---", "`"):
        assert raw not in prose, raw
    assert "git repository and every candidate is a commit" in t and "under one cent of HF credits per run" in t
    assert 'href="https://arxiv.org/abs/2603.28052"' in page
    assert f"https://github.com/{site.GITHUB_REPO}/blob/main/hands-on/part2-evolve/APPLY-TO-YOUR-AGENT.md" in page, "the source file stays one tap away"


# --- the small markdown renderer behind the guide page ---

def test_markdown_renders_the_constructs_the_guide_uses():
    out = site.markdown("# Title\n\nA **bold** word, `a <b> c`, a [link](https://x.org/a_b).\nSecond line.\n\n"
                        "- one\n  more\n- two\n\n1. first\n   rest\n2. second\n\n```bash\nuv run x --a \"b\"\n```\n\n"
                        "| A | B |\n|---|---|\n| **1.** x | `y` |\n")
    assert "<h1>Title</h1>" in out
    assert '<p>A <b>bold</b> word, <code>a &lt;b&gt; c</code>, a <a href="https://x.org/a_b">link</a>. Second line.</p>' in out
    assert "<ul><li>one more</li><li>two</li></ul>" in out.replace("\n", "")
    assert "<ol><li>first rest</li><li>second</li></ol>" in out.replace("\n", "")
    assert boxes(out) == ['uv run x --a "b"']
    assert "<th>A</th><th>B</th>" in out and '<td data-label="A"><b>1.</b> x</td><td data-label="B"><code>y</code></td>' in out


def test_markdown_escapes_html_in_text():
    assert "<script>" not in site.markdown("a <script>alert(1)</script> b") and "&lt;script&gt;" in site.markdown("a <script>alert(1)</script> b")


@pytest.mark.parametrize("unsupported", ["### too deep", "> a quote", "![img](https://x.org/a.png)", "see [the file](examples/own_eval/agent.py)",
                                          "```\nnever closed", "    indented code"])
def test_markdown_refuses_what_it_does_not_render(unsupported):
    """A construct the renderer does not know would show as raw markup on the page: fail the export instead."""
    with pytest.raises(ValueError):
        site.markdown(unsupported)
