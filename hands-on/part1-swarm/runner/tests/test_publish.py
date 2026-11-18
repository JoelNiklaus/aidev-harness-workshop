"""devtools/publish.py: the publisher attaches the wheel to the public repo's release for this version."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("publish", Path(__file__).parents[1] / "devtools" / "publish.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class FakeGh:
    """Records gh calls; `exists` says whether the release is already there."""

    def __init__(self, exists=False):
        self.exists, self.calls = exists, []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[:2] == ["release", "view"]:
            return 0 if self.exists else 1
        return 0


def wheel(tmp_path, version="0.2.0"):
    w = tmp_path / f"aidev_swarm-{version}-py3-none-any.whl"
    w.write_bytes(b"PK")
    return w


def test_creates_the_release_with_the_wheel_when_it_is_new(tmp_path):
    gh = FakeGh(exists=False)
    url = publish.publish(wheel(tmp_path), "0.2.0", "owner/repo", gh=gh)
    create = [c for c in gh.calls if c[:2] == ["release", "create"]]
    assert len(create) == 1 and "swarm-v0.2.0" in create[0] and str(wheel(tmp_path)) in create[0]
    assert "--repo" in create[0] and "owner/repo" in create[0]
    assert url == "https://github.com/owner/repo/releases/download/swarm-v0.2.0/aidev_swarm-0.2.0-py3-none-any.whl"


def test_replaces_the_asset_when_the_release_exists(tmp_path):
    gh = FakeGh(exists=True)
    publish.publish(wheel(tmp_path), "0.2.0", "owner/repo", gh=gh)
    upload = [c for c in gh.calls if c[:2] == ["release", "upload"]]
    assert len(upload) == 1 and "--clobber" in upload[0] and not any(c[:2] == ["release", "create"] for c in gh.calls)


def test_published_url_is_the_one_launch_installs(tmp_path):
    from aidev_swarm import launch

    url = publish.publish(wheel(tmp_path), "0.2.0", launch.GITHUB_REPO, gh=FakeGh())
    assert url == launch.default_package_url("0.2.0")


def test_readme_wheel_url_is_the_one_launch_installs():
    """The README line is what the coordinator's /join page and the deck show attendees (they parse it)."""
    import re

    from aidev_swarm import launch

    readme = (Path(__file__).parents[1] / "README.md").read_text()
    shown = re.search(r"`<WHEEL_URL>` is today `([^`]+)`", readme)
    assert shown, "README keeps the line `<WHEEL_URL>` is today `<url>` (coordinator join.py and the deck read it)"
    assert shown.group(1) == launch.default_package_url()


def test_readme_names_the_default_coordinator():
    from aidev_swarm import cli

    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert f"The coordinator defaults to `{cli.DEFAULT_COORDINATOR}`" in readme
