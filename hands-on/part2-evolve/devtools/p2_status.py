"""Status of Phase 2 experiment runs: one line per iteration, from each run folder's state.json.

    uv run python devtools/p2_status.py runs/p2/A1 runs/p2/A2 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def main(dirs: list[str]) -> int:
    for d in dirs:
        p = Path(d) / "state.json"
        if not p.exists():
            print(f"{d}: no state yet")
            continue
        st = json.loads(p.read_text())
        print(f"{d}: iteration {st['iteration']}, best {st['incumbent']}, margin {pct(st['margin'])}, flip {pct(st['flip_rate'])}, "
              f"spend ${st['spend_usd']:.2f} (proposer API-equivalent ${st['proposer_usd']:.2f}), curve {[round(100 * c) for c in st['curve']]}")
        for h in st["history"]:
            first = f" first {pct(h.get('val_parent_first'))}->{pct(h.get('val_child_first'))}" if "val_child_first" in h else ""
            print(f"  {h['iteration']:>2} {h['candidate']} {str(h.get('title'))[:48]:48} {pct(h.get('val_parent')):>4}->{pct(h.get('val_child')):>4}{first} "
                  f"{h['verdict'][:36]:36} {h.get('iter_s', 0) / 60:4.1f} min ${h.get('iter_usd', 0):.3f} prop {h.get('proposer_s', 0):.0f}s")
        ho = st.get("heldout")
        if ho:
            print(f"  held-out {ho['n_tasks']} tasks x {ho['trials']}: seed {pct(ho['seed'])} -> {ho['best_candidate']} {pct(ho['best'])}, "
                  f"diff {100 * ho.get('diff', 0):+.0f} pts, 95% CI [{100 * ho.get('ci_low', 0):+.0f}, {100 * ho.get('ci_high', 0):+.0f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
