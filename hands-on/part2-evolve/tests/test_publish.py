"""devtools/publish.py: the wheel becomes an asset of the public repo's release for this version; README shows its URL."""

import importlib.util
import re
from pathlib import Path

from aidev_evolve import launch

spec = importlib.util.spec_from_file_location("evolve_publish", Path(__file__).parents[1] / "devtools" / "publish.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class FakeGh:
    """Records gh calls; `exists` says whether the release is already there."""

    def __init__(self, exists=False, rc=0):
        self.exists, self.rc, self.calls = exists, rc, []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[:2] == ["release", "view"]:
            return 0 if self.exists else 1
        return self.rc


def wheel(tmp_path, version="0.9.0"):
    w = tmp_path / f"aidev_evolve-{version}-py3-none-any.whl"
    w.write_bytes(b"PK")
    return w


def test_creates_the_release_with_the_wheel_when_it_is_new(tmp_path):
    gh = FakeGh(exists=False)
    url = publish.publish(wheel(tmp_path), "0.9.0", "owner/repo", gh=gh)
    create = [c for c in gh.calls if c[:2] == ["release", "create"]]
    assert len(create) == 1 and "evolve-v0.9.0" in create[0] and str(wheel(tmp_path)) in create[0]
    assert "owner/repo" in create[0]
    assert url == "https://github.com/owner/repo/releases/download/evolve-v0.9.0/aidev_evolve-0.9.0-py3-none-any.whl"


def test_replaces_the_asset_when_the_release_exists(tmp_path):
    gh = FakeGh(exists=True)
    publish.publish(wheel(tmp_path), "0.9.0", "owner/repo", gh=gh)
    upload = [c for c in gh.calls if c[:2] == ["release", "upload"]]
    assert len(upload) == 1 and "--clobber" in upload[0] and not any(c[:2] == ["release", "create"] for c in gh.calls)


def test_a_failing_gh_stops_the_publish(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="gh failed"):
        publish.publish(wheel(tmp_path), "0.9.0", "owner/repo", gh=FakeGh(rc=1))


def test_published_url_is_the_one_the_jobs_install(tmp_path):
    assert publish.publish(wheel(tmp_path), "0.9.0", launch.GITHUB_REPO, gh=FakeGh()) == launch.default_package_url("0.9.0")


def test_readme_wheel_url_is_the_one_the_jobs_install():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    shown = re.search(r"`<WHEEL_URL>` is today `([^`]+)`", readme)
    assert shown, "README keeps the line `<WHEEL_URL>` is today `<url>`"
    assert shown.group(1) == launch.default_package_url()
