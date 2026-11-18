import hashlib

import pytest

from aidev_evolve import cli


def test_room_kit_creates_offline_shared_seed_and_never_overwrites(tmp_path, capsys):
    work = tmp_path / 'room'
    assert cli.main(['room-init', str(work)]) == 0
    base = (work / 'base.py').read_bytes()
    assert (work / 'agent.py').read_bytes() == base
    assert hashlib.sha256(base).hexdigest() in (work / 'BASE.sha256').read_text()
    assert 'recorded' in (work / 'README.md').read_text().lower()
    (work / 'agent.py').write_text('my changes')
    with pytest.raises(SystemExit):
        cli.main(['room-init', str(work)])
    assert (work / 'agent.py').read_text() == 'my changes'


def test_room_diff_prints_patch_without_running_it_and_refuses_changed_base(tmp_path, capsys):
    work = tmp_path / 'room'
    cli.main(['room-init', str(work)])
    marker = tmp_path / 'executed'
    with (work / 'agent.py').open('a') as f:
        f.write(f'\nopen({str(marker)!r}, "w").write("bad")\n')
    assert cli.main(['room-diff', str(work)]) == 0
    output = capsys.readouterr().out
    assert '+open(' in output
    assert not marker.exists()
    (work / 'base.py').write_text('changed base')
    with pytest.raises(SystemExit, match='base'):
        cli.main(['room-diff', str(work)])


def test_replay_is_clearly_labeled_as_recorded(capsys, monkeypatch):
    from aidev_evolve import replay
    monkeypatch.setattr(replay, 'replay', lambda *args, **kwargs: print('historical events'))
    assert cli.main(['replay']) == 0
    assert 'recorded' in capsys.readouterr().out.lower()
