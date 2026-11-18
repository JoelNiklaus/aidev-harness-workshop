"""Stand-in for the coordinator's /evolve board (PROTOCOL.md), stdlib only.

    python devtools/fake_board.py --port 8766        # then: aidev-evolve run --board http://127.0.0.1:8766

It accepts any bearer token and uses it as the user name's seed (no Hub call). The coordinator owner can
lift `Store.state()` (the aggregation) and `PAGE` (race chart + patch feed) into the real Space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Store:
    def __init__(self) -> None:
        self._events: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        self._order: list[tuple[str, str, str, int]] = []
        self._lock = threading.Lock()

    def add(self, user: str, event: dict[str, Any]) -> None:
        key = (user, event["run_id"], event["kind"], int(event.get("iteration") or 0))
        with self._lock:
            if key not in self._events:
                self._order.append(key)
            self._events[key] = {**event, "user": user, "received_at": time.time()}

    def state(self) -> dict[str, Any]:
        with self._lock:
            events = [self._events[k] for k in self._order]
        runs: dict[str, dict[str, Any]] = {}
        feed = []
        for e in events:
            r = runs.setdefault(e["run_id"], {"run_id": e["run_id"], "user": e["user"], "curve": [], "spend_usd": 0.0, "kept": 0, "iterations": 0, "heldout": None})
            r["spend_usd"] = max(r["spend_usd"], float(e.get("spend_usd") or 0.0))
            if e["kind"] == "seed":
                r["curve"] = [e["val_score"]] + r["curve"][1:]
            elif e["kind"] == "iteration":
                r["iterations"] += 1
                r["kept"] += bool(e.get("accepted"))
                r["curve"].append(e["best_val"])
                feed.append({k: e.get(k) for k in ("user", "run_id", "iteration", "title", "accepted", "verdict", "val_child", "val_parent", "fixed")})
            elif e["kind"] == "heldout":
                r["heldout"] = {k: e.get(k) for k in ("seed", "best", "n_tasks", "trials")}
        feed.reverse()
        rs = list(runs.values())
        totals = {"runs": len(rs), "iterations": sum(r["iterations"] for r in rs), "kept": sum(r["kept"] for r in rs),
                  "spend_usd": round(sum(r["spend_usd"] for r in rs), 4)}
        return {"runs": rs, "feed": feed[:30], "totals": totals}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Evolve the harness together</title>
<style>body{background:#111;color:#eee;font:26px system-ui;margin:24px}h1{font-size:40px;margin:0 0 12px}
.row{display:flex;gap:32px}.feed{flex:1;max-width:40%}.item{margin:10px 0}.kept{color:#7CFC9A}.drop{color:#888}
#totals{font-size:28px;color:#bbb}svg text{fill:#ddd;font-size:22px}</style></head><body>
<h1>Evolve the harness: GLM-5.3-Flash on Terminal-Bench 2.1</h1><div id="totals"></div>
<div class="row"><svg id="race" width="1100" height="640" aria-label="race chart"></svg><div class="feed" id="feed"></div></div>
<script>
const W=1100,H=640,L=80,B=60,T=20,R=160;
function draw(s){
  const runs=s.runs, n=Math.max(4,...runs.map(r=>r.curve.length-1));
  const x=i=>L+i*(W-L-R)/n, y=v=>T+(1-v)*(H-T-B);
  let g='';
  for(let v=0;v<=1.0001;v+=0.25){g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#333"/><text x="10" y="${y(v)+8}">${Math.round(v*100)}%</text>`}
  for(let i=0;i<=n;i++){g+=`<text x="${x(i)-6}" y="${H-20}">${i}</text>`}
  const top=[...runs].sort((a,b)=>b.curve.at(-1)-a.curve.at(-1)).slice(0,10).map(r=>r.run_id);
  runs.forEach((r,k)=>{const c=`hsl(${(k*67)%360} 70% 60%)`;
    g+=`<polyline fill="none" stroke="${c}" stroke-width="4" points="${r.curve.map((v,i)=>x(i)+','+y(v)).join(' ')}"/>`;
    if(top.includes(r.run_id)){const i=r.curve.length-1;g+=`<text x="${x(i)+8}" y="${y(r.curve[i])+8}" style="fill:${c}">${r.user}</text>`}});
  document.getElementById('race').innerHTML=g;
  document.getElementById('feed').innerHTML='<b>Patches</b>'+s.feed.map(f=>`<div class="item ${f.accepted?'kept':'drop'}">${f.user}: ${f.title||''} ${f.accepted?'+'+Math.round(100*(f.val_child-f.val_parent))+' pts':'(dropped)'}</div>`).join('');
  const t=s.totals;document.getElementById('totals').textContent=`${t.runs} runs, ${t.iterations} iterations, ${t.kept} patches kept, $${t.spend_usd.toFixed(2)} of HF credits`;
}
async function tick(){try{draw(await (await fetch('/api/evolve/state')).json())}catch(e){} setTimeout(tick,5000)}
tick();
</script></body></html>"""


def serve(port: int = 8766, host: str = "127.0.0.1", store: Store | None = None) -> ThreadingHTTPServer:
    store = store or Store()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            return None

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/evolve/state":
                self._send(200, json.dumps(store.state()).encode())
            elif self.path in ("/evolve", "/"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"{}")

        def do_POST(self) -> None:  # noqa: N802
            auth = self.headers.get("Authorization", "")
            if self.path != "/api/evolve/submit" or not auth.startswith("Bearer "):
                self._send(401 if self.path == "/api/evolve/submit" else 404, b"{}")
                return
            try:
                event = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                user = (event.get("run_id") or "").split("-")[0] or "user-" + hashlib.sha256(auth.encode()).hexdigest()[:6]
                store.add(user, event)
            except (ValueError, KeyError):
                self._send(422, b"{}")
                return
            self._send(200, b'{"ok": true}')

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"evolve board stand-in on http://{a.host}:{a.port}/evolve")
    serve(a.port, a.host).serve_forever()


if __name__ == "__main__":
    main()
