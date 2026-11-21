"""Short links: static redirect pages served by GitHub Pages from the public repo's docs/ folder, so the slides can show
a URL people can type (the board host alone has 43 characters)."""

from __future__ import annotations

import re

from swarm_coordinator import deploy, join, shortlinks

BOARD = deploy.space_url(deploy.DEFAULT_SPACE)


def target(html: str) -> str:
    return re.search(r'<meta http-equiv="refresh" content="0; url=([^"]+)">', html).group(1)


def test_1_opens_the_part1_join_page_and_2_the_part2_start_page_of_the_workshop_board():
    pages = shortlinks.pages()
    assert target(pages["1/index.html"]) == BOARD + join.PATH
    assert target(pages["2/index.html"]) == BOARD + join.EVOLVE_PATH


def test_guide_opens_the_apply_to_your_agent_guide_in_the_public_repo():
    t = target(shortlinks.pages()["guide/index.html"])
    assert t == "https://github.com/JoelNiklaus/aidev-harness-workshop/blob/main/hands-on/part2-evolve/APPLY-TO-YOUR-AGENT.md"
    assert (deploy.COORD.parents[1] / "part2-evolve" / "APPLY-TO-YOUR-AGENT.md").is_file(), "the guide the link opens exists in this repo"


def test_every_redirect_also_works_without_meta_refresh():
    """Some in-app browsers ignore the meta tag: a script and a plain link carry the same address."""
    for path, html in shortlinks.pages().items():
        if path in ("index.html", ".nojekyll"):
            continue
        t = target(html)
        assert f'location.replace("{t}")' in html and f'<a href="{t}">' in html and '<link rel="canonical" href="' + t in html


def test_the_index_lists_every_short_link_with_its_public_address():
    index = shortlinks.pages()["index.html"]
    for name in shortlinks.LINKS:
        assert f"{shortlinks.PUBLIC_BASE}/{name}" in index


def test_the_public_base_is_the_public_repos_pages_address():
    from swarm_coordinator import paths

    launch = (paths.runner_dir() / "src" / "aidev_swarm" / "launch.py").read_text()
    assert f'GITHUB_REPO = "{shortlinks.GITHUB_REPO}"' in launch  # the repo the wheels are released from
    assert shortlinks.PUBLIC_BASE == "https://niklaus.ai/aidev-harness-workshop"
    assert shortlinks.PUBLIC_BASE.rsplit("/", 1)[1] == shortlinks.GITHUB_REPO.split("/")[1]  # a project site lives under the repo name


def test_write_puts_the_pages_under_the_folder_and_tells_pages_not_to_run_jekyll(tmp_path):
    shortlinks.write(tmp_path / "docs")
    names = {p.relative_to(tmp_path / "docs").as_posix() for p in (tmp_path / "docs").rglob("*") if p.is_file()}
    assert names == {"index.html", ".nojekyll", "1/index.html", "2/index.html", "guide/index.html"}
