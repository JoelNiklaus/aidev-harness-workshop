"""scripts/evolve_space_check.py: the live /evolve check (pure parts; the Hub and HTTP calls are not run here)."""

import importlib.util
from pathlib import Path

from swarm_coordinator import evolve, paths
from swarm_coordinator.evolve_replay import load_recorded

spec = importlib.util.spec_from_file_location("evolve_space_check", Path(__file__).parents[1] / "scripts" / "evolve_space_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_check_events_are_a_short_real_loop_under_a_check_run_id():
    evs = check.check_events(load_recorded(paths.evolve_replay_path()), "check-123")
    assert [e["kind"] for e in evs] == ["seed", "iteration", "iteration", "heldout"]
    assert {e["run_id"] for e in evs} == {"check-123"}
    for e in evs:
        # what the part2 client sends: its Board.submit adds the schema
        evolve.validate({"schema": evolve.SCHEMA, **e})


def test_problems_name_what_the_board_lacks():
    evs = check.check_events(load_recorded(paths.evolve_replay_path()), "check-123")
    store = evolve.EvolveStore()
    for e in evs:
        store.add("joel", evolve.validate({"schema": evolve.SCHEMA, **e}))
    state = evolve.build_state(store.events(), reference=[{"label": "recorded B1", "curve": [[0, 0.4]]}])
    assert check.problems(state, "joel", "check-123") == []
    empty = evolve.build_state([], reference=[])
    assert check.problems(empty, "joel", "check-123") == ["run check-123 of joel is not on the board", "no recorded backdrop"]
