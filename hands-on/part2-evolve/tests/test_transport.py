"""Records travel from the HF Job to the laptop as marked lines in the Job's log."""

from aidev_evolve import transport


def test_roundtrip_through_log_lines_mixed_with_other_output():
    big = {"task": "fix-git", "trajectory": {"steps": [{"output": "x" * 50_000 + str(i)} for i in range(20)]}}
    small = {"task": "regex-log", "passed": True}
    lines = transport.encode(big, msg_id="a1", chunk_chars=4000) + ["harbor: noise line"] + transport.encode(small, msg_id="a2")
    assert all(len(line) <= 4100 for line in transport.encode(big, msg_id="a1", chunk_chars=4000))
    dec = transport.Decoder()
    got = [obj for line in lines for obj in dec.feed(line)]
    assert got == [big, small]


def test_decoder_ignores_repeated_lines_and_returns_each_message_once():
    lines = transport.encode({"n": 1}, msg_id="m", chunk_chars=10)
    dec = transport.Decoder()
    got = [obj for line in lines + lines for obj in dec.feed(line)]
    assert got == [{"n": 1}]


def test_decoder_tolerates_log_prefixes_and_trailing_newlines():
    [line] = transport.encode({"ok": True}, msg_id="p")
    dec = transport.Decoder()
    assert dec.feed("2026-11-17T12:00:00Z " + line + "\n") == [{"ok": True}]


def test_done_marker_is_recognised():
    assert transport.is_done(transport.done_line(3))
    assert transport.done_count(transport.done_line(3)) == 3
    assert not transport.is_done("DONE but not ours")
