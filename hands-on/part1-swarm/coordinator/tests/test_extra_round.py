from conftest import PRICES, make_result
from swarm_coordinator.store import Store
from swarm_coordinator.state import build_state


def test_late_runner_gets_fastest_extra_and_restart_counts_it(grid, sink, clock):
    s = Store(grid, PRICES, clock=clock, sink=sink, extra_seconds=900)
    base = [s.claim(str(i))[0] for i in range(len(grid.cells))]
    b = base[1]
    s.submit(b["cell_id"], "1", make_result(b["cell_id"], wall_s=20))
    x = s.claim("late")[0]
    assert x["trial"] == grid.trials_per_cell and x["task"] == b["task"]
    s.submit(x["cell_id"], "late", make_result(x["cell_id"]))
    r = Store(grid, PRICES, clock=clock, sink=sink, extra_seconds=900)
    r.restore()
    state = build_state(grid, r.snapshot(), [], {}, mode="live")
    assert state["counts"]["base_done"] == 1 and state["counts"]["extra_done"] == 1
    assert sum(h["room"]["n"] for h in state["harnesses"] if h["room"]) == 2
    clock.advance(901)
    assert not r.claim("end")


def test_restart_keeps_original_stop_even_if_runner_clock_is_wrong(grid, sink, clock):
    s = Store(grid, PRICES, clock=clock, sink=sink, extra_seconds=100)
    cid = s.claim("first")[0]["cell_id"]
    clock.advance(80)
    s.submit(cid, "first", make_result(cid, started_at="2001-01-01T00:00:00Z"))
    r = Store(grid, PRICES, clock=clock, sink=sink, extra_seconds=100)
    r.restore()
    assert r.claim("late")
    clock.advance(21)
    assert not r.claim("late-again")
