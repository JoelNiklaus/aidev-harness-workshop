"""GET /evolve/join: the Part 2 start page on the board (the attendee site's static /2 shows the same commands): one command per seating block, with Copy buttons.

The wheel URL is the `<WHEEL_URL>` line of the Part 2 README (pinned there to the package version) and the block ->
strategy table is the Part 2 package's own (`aidev_evolve.strategy.BLOCKS`), so page, README and CLI cannot drift."""

from __future__ import annotations

import ast
import re

import pytest
from fastapi.testclient import TestClient

from swarm_coordinator import join, paths
from swarm_coordinator.app import create_app
from swarm_coordinator.deploy import stage

from test_join import boxes

EVOLVE = paths.evolve_dir()


def evolve_url() -> str:
    return re.search(r"`<WHEEL_URL>` is today `([^`]+)`", (EVOLVE / "README.md").read_text()).group(1)


@pytest.fixture
def page(store):
    r = TestClient(create_app(store, auth=None, reference=[], meta={})).get("/evolve/join")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    return r.text


def test_the_blocks_are_the_part2_packages_blocks():
    src = (EVOLVE / "src" / "aidev_evolve" / "strategy.py").read_text()
    assert join.BLOCKS == ast.literal_eval(re.search(r"^BLOCKS = (\{.*\})$", src, re.M).group(1))


def test_the_page_gives_the_measured_run_time(page):
    """Measured for the 0.6.0 workshop run (23 Nov 2026): about 6 min alone; launch to held-out result with 40 runs at
    once took 12.2 min at the median and 15.0 at p90. The earlier "about 20 minutes" was the long loop before 0.6.0."""
    assert join.EVOLVE_RUN_TIME == "One run takes about 6 minutes on its own and 12 to 15 minutes during the session."
    assert join.EVOLVE_RUN_TIME in page and "20 minutes" not in page


def test_the_run_time_on_its_own_is_the_one_aidev_evolve_strategies_prints():
    """`aidev-evolve strategies` prints "about N min when run alone" for a one-Job workshop run (every block's strategy):
    N comes from these two constants of the Part 2 package, and the page's sentence names the same N."""
    src = (EVOLVE / "src" / "aidev_evolve" / "strategy.py").read_text()
    propose, job = (float(x) for x in re.search(r"(?m)^WORKSHOP_PROPOSE_MIN, WORKSHOP_JOB_MIN = ([\d.]+), ([\d.]+)", src).groups())
    assert "minutes = WORKSHOP_PROPOSE_MIN + WORKSHOP_JOB_MIN" in src and "about {est['minutes']:.0f} min when run alone" in src
    assert f"about {propose + job:.0f} minutes on its own" in join.EVOLVE_RUN_TIME


def test_each_block_has_its_run_command_verbatim_for_claude_and_for_codex(page):
    got = boxes(page)
    for block in join.BLOCKS:
        assert got[f"run-{block}"] == f'uvx --from "{evolve_url()}" aidev-evolve run --block {block}'
        assert got[f"run-{block}-codex"] == f'uvx --from "{evolve_url()}" aidev-evolve run --block {block} --proposer codex'


def test_the_page_names_each_blocks_strategy_next_to_its_command(page):
    for block, preset in join.BLOCKS.items():
        section = page[page.index(f'id="block-{block}"'):]
        section = section[:section.index("</section>")]
        assert preset in section and f"run-{block}" in section and block in section.lower()


def test_the_page_has_the_check_the_final_agent_and_the_offline_replay(page):
    got = boxes(page)
    assert got["check"] == f'uvx --from "{evolve_url()}" aidev-evolve check'
    assert got["show"] == f'uvx --from "{evolve_url()}" aidev-evolve show'
    assert got["replay"] == f'uvx --from "{evolve_url()}" aidev-evolve replay'


def test_every_box_has_a_copy_button_and_the_page_links_the_board(page):
    targets = re.findall(r'<button[^>]*data-copy="([\w-]+)"', page)
    assert set(targets) == set(boxes(page)) and len(targets) == len(set(targets))
    assert re.search(r'<a [^>]*href="/evolve"', page) and 'name="viewport"' in page
    assert not re.search(r"\bthe room\b|\broom's\b", page, re.I)  # Joel: never an actor or a label for attendees


def test_one_switch_shows_the_codex_commands_instead_of_the_claude_ones(page):
    """Both sets are in the page; a body class picks one, so Copy always copies what is shown."""
    assert re.search(r"body\.codex \.for-claude\s*\{\s*display:\s*none", page) and re.search(r"body:not\(\.codex\) \.for-codex\s*\{\s*display:\s*none", page)
    assert len(re.findall(r'class="box for-codex"', page)) == len(join.BLOCKS) == len(re.findall(r'class="box for-claude"', page))
    assert 'data-agent="codex"' in page and 'data-agent="claude"' in page


def test_the_page_never_says_which_strategy_wins(page):
    """The prediction comes before the reveal (expert review): no ranking, no default hint."""
    assert not re.search(r"winner|ranks? first|best strategy|default", page, re.I)


def test_staged_space_renders_the_same_start_page(tmp_path):
    out = stage(tmp_path / "space")
    assert (out / "evolve" / "README.md").is_file()
    assert join.evolve_page(out / "evolve" / "README.md") == join.evolve_page(EVOLVE / "README.md")


def test_the_start_page_fails_loudly_without_the_wheel_url(tmp_path):
    (tmp_path / "README.md").write_text("# no wheel line\n")
    with pytest.raises(ValueError, match="WHEEL_URL"):
        join.evolve_page(tmp_path / "README.md")
