"""GET /join: the page attendees open from the slide's QR code to copy the launch command and the agent prompt.

The command and the prompt come from the runner's own docs (runner/README.md, runner/AGENT_PROMPT.md), so
the page, the README and the deck (which reads the same README line through `join.wheel_url`) cannot drift.
"""

from __future__ import annotations

import html
import re
import shutil

import pytest
from fastapi.testclient import TestClient

from swarm_coordinator import join, paths
from swarm_coordinator.app import create_app
from swarm_coordinator.deploy import stage

RUNNER = paths.runner_dir()


def readme_url() -> str:
    return re.search(r"`<WHEEL_URL>` is today `([^`]+)`", (RUNNER / "README.md").read_text()).group(1)


@pytest.fixture
def client(store):
    return TestClient(create_app(store, auth=None, reference=[], meta={}))


def boxes(page: str) -> dict[str, str]:
    """id -> text of every copyable box (unescaped, as the clipboard gets it)."""
    return {m[1]: html.unescape(m[2]) for m in re.finditer(r'<pre id="([\w-]+)"[^>]*>(.*?)</pre>', page, re.S)}


def test_wheel_url_is_the_line_of_the_runner_readme():
    assert join.wheel_url((RUNNER / "README.md").read_text()) == readme_url()
    assert readme_url().endswith(".whl")


def test_launch_command_is_the_uvx_line_attendees_run():
    assert join.launch_command("https://x/a.whl") == 'uvx --from "https://x/a.whl" aidev-swarm launch'


def test_join_page_is_public_html_with_the_exact_command_and_prompt(client):
    r = client.get(join.PATH)  # no Authorization header: like the board, /join needs none
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    got = boxes(r.text)
    command = f'uvx --from "{readme_url()}" aidev-swarm launch'
    assert command in got.values(), "the launch command sits in its own box, verbatim"
    prompt = [t for t in got.values() if t.startswith("Run ") and "aidev-swarm launch" in t]
    assert len(prompt) == 1, "the agent prompt sits in its own box"
    assert command in prompt[0] and "<WHEEL_URL>" not in prompt[0]


def test_prompt_box_is_the_whole_agent_prompt_from_the_runner():
    quoted = [ln.removeprefix("> ") for ln in (RUNNER / "AGENT_PROMPT.md").read_text().splitlines() if ln.startswith("> ")]
    expected = " ".join(quoted).replace("<WHEEL_URL>", readme_url())
    assert join.agent_prompt((RUNNER / "AGENT_PROMPT.md").read_text(), readme_url()) == expected


def test_join_page_has_three_steps_copy_buttons_and_a_board_link(client):
    page = client.get(join.PATH).text
    steps = re.findall(r"<h2[^>]*>(.*?)</h2>", page, re.S)
    assert len(steps) == 3
    assert "uv" in steps[0] and "token" in steps[1].lower() and "agent" in steps[2].lower()
    ids = boxes(page)
    targets = re.findall(r'<button[^>]*data-copy="([\w-]+)"', page)
    assert set(targets) == set(ids), "every box has one Copy button and every button copies an existing box"
    assert "uvx hf auth login" in ids.values()
    for permission in ("Make calls to Inference Providers", "Start and manage Jobs"):
        assert permission in page and permission in (RUNNER / "README.md").read_text(), "same token advice as the runner README (the prep email copies it)"
    assert re.search(r'<a [^>]*href="/"', page), "a link back to the live board"
    assert 'name="viewport"' in page, "phone-friendly"


def test_join_page_fails_loudly_without_the_wheel_url(tmp_path):
    shutil.copy2(RUNNER / "AGENT_PROMPT.md", tmp_path / "AGENT_PROMPT.md")
    (tmp_path / "README.md").write_text("# runner\n\nno wheel line here\n")
    with pytest.raises(ValueError, match="WHEEL_URL"):
        join.page(tmp_path)


def test_join_page_fails_loudly_when_the_prompt_lost_the_command(tmp_path):
    shutil.copy2(RUNNER / "README.md", tmp_path / "README.md")
    (tmp_path / "AGENT_PROMPT.md").write_text("> Run something else.\n")
    with pytest.raises(ValueError, match="aidev-swarm launch"):
        join.page(tmp_path)


def test_staged_space_renders_the_same_page(tmp_path):
    out = stage(tmp_path / "space")
    assert join.page(out / "runner") == join.page(RUNNER)


def test_join_page_never_names_the_room(client):
    # Joel: "the room" is not an actor or a label on anything attendees read
    page = client.get("/join").text
    assert not re.search(r"\bthe room\b|\broom's\b", page, re.I)
