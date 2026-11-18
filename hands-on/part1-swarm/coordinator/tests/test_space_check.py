"""scripts/space_check.py: the live check of a deployed Space (pure parts; the Hub and HTTP calls are not run here)."""

import importlib.util
import sys
from pathlib import Path

from swarm_coordinator import grid as grid_mod, paths
from swarm_coordinator.replay import load_results
from swarm_coordinator.state import build_state
from swarm_coordinator.store import Store, validate

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("space_check", Path(__file__).parents[1] / "scripts" / "space_check.py")
space_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space_check)


def store():
    import json

    g = grid_mod.Grid.load(paths.grid_path())
    return Store(g, json.loads(paths.prices_path().read_text()))


def test_synthetic_result_passes_the_coordinators_schema_and_says_what_it_is():
    s = store()
    claim = s.claim("checker")[0]
    r = space_check.synthetic_result(claim, "checker")
    validate(r)
    assert r["cell_id"] == claim["cell_id"] and r["hf_user"] == "checker"
    assert "space_check" in r["error"] and r["n_calls"] == 0


def test_state_problems_accepts_a_clean_board_and_names_every_deviation():
    import json

    s = store()
    ref = load_results(paths.reference_path())
    clean = build_state(s.grid, s.snapshot(), ref, {}, mode="live")
    assert space_check.state_problems(clean, done=0, grid=s.grid) == []
    dirty = json.loads(json.dumps(clean))
    dirty["counts"]["done"] = 1
    dirty["harnesses"][0]["reference"] = None
    problems = space_check.state_problems(dirty, done=0, grid=s.grid)
    assert any("done" in p for p in problems) and any("reference" in p for p in problems)


def test_the_live_check_claims_as_a_current_runner():
    from swarm_coordinator.app import runner_ok

    assert runner_ok(space_check.USER_AGENT)
