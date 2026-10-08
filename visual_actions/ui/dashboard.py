"""Minimal live dashboard: a local HTTP page with a server-sent-events stream.

Standard library only, so it is identical on every platform. Bus handlers run on the
main thread and only append to in-memory state; the HTTP server runs on a daemon
thread and reads it under a lock.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..core.config import Config
from ..core.drag import DragEvent
from ..core.events import ActionFired, Bus, HoldProgress, ModeChanged, SnapPreview, TokenEmitted

LOG_SIZE = 40
StatsFn = Callable[[], dict[str, Any]]


class Dashboard:
    def __init__(self, bus: Bus, cfg: Config, stats: StatsFn | None = None, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.cfg = cfg
        self.stats = stats or (dict)
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []
        self.state: dict[str, Any] = {
            "mode": "idle",
            "namespace": None,
            "hold": 0.0,
            "token": None,
            "token_conf": 0.0,
            "token_still": None,
            "tokens_per_s": 0.0,
            "snap": None,
            "last_action": None,
        }
        self.log: deque[dict[str, Any]] = deque(maxlen=LOG_SIZE)
        self._token_times: deque[float] = deque(maxlen=60)
        bus.subscribe(ModeChanged, self._on_mode)
        bus.subscribe(HoldProgress, self._on_hold)
        bus.subscribe(TokenEmitted, self._on_token)
        bus.subscribe(ActionFired, self._on_action)
        bus.subscribe(DragEvent, self._on_drag)
        bus.subscribe(SnapPreview, self._on_snap)
        self.server = ThreadingHTTPServer((host, port), _handler_for(self))
        self.port = self.server.server_address[1]
        self.url = f"http://{host}:{self.port}"
        threading.Thread(target=self.server.serve_forever, name="dashboard", daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()

    # -- bus handlers (main thread) ------------------------------------------------

    def _emit(self, kind: str, **fields: Any) -> None:
        entry = {"t": time.time(), "kind": kind, **fields}
        with self._lock:
            self.log.appendleft(entry)
            for q in self._clients:
                try:
                    q.put_nowait(entry)
                except queue.Full:
                    pass

    def _on_mode(self, ev: ModeChanged) -> None:
        with self._lock:
            self.state["mode"], self.state["namespace"] = ev.new, ev.namespace
            if ev.new != "holding":
                self.state["hold"] = 0.0
        self._emit("mode", old=ev.old, new=ev.new)

    def _on_hold(self, ev: HoldProgress) -> None:
        with self._lock:
            self.state["hold"] = round(ev.fraction, 3)
        self._emit("hold", fraction=round(ev.fraction, 3))

    def _on_token(self, ev: TokenEmitted) -> None:
        now = time.time()
        self._token_times.append(now)
        recent = [t for t in self._token_times if now - t <= 5]
        with self._lock:
            self.state.update(
                token=ev.token.name,
                token_conf=round(ev.token.confidence, 2),
                token_still=ev.token.still,
                tokens_per_s=round(len(recent) / 5, 1),
            )
        self._emit("token", name=ev.token.name, conf=round(ev.token.confidence, 2), still=ev.token.still)

    def _on_action(self, ev: ActionFired) -> None:
        with self._lock:
            self.state["last_action"] = ev.action.name
        self._emit("action", name=ev.action.name, ok=ev.ok, message=ev.message)

    def _on_drag(self, ev: DragEvent) -> None:
        if ev.phase.value == "move":
            return  # 30 Hz, not for the log
        self._emit("drag", phase=ev.phase.value, window=ev.window, snapped=ev.snapped)

    def _on_snap(self, ev: SnapPreview) -> None:
        with self._lock:
            self.state["snap"] = ev.zone

    # -- http side ----------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self.state)
            log = list(self.log)
        bindings = [
            {"namespace": ns, "gesture": b["gesture"], "action": b["action"].get("name", b["action"].get("kind"))}
            for ns, nc in self.cfg.namespaces.items()
            for b in nc.bindings
        ]
        return {
            "state": state,
            "log": log,
            "bindings": bindings,
            "timing": asdict(self.cfg.timing),
            "drag": {k: v for k, v in asdict(self.cfg.drag).items() if k in ("enabled", "pinch_on", "pinch_off", "snap_enabled", "gain", "depth_gain", "ref_hand_scale")},
            "stats": self.stats(),
            "uptime_s": round(time.time() - self.started_at),
        }

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)


def _handler_for(dash: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state.json":
                self._send(200, json.dumps(dash.snapshot()).encode(), "application/json")
            elif self.path == "/events":
                self._stream()
            elif self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q = dash.subscribe()
            try:
                self.wfile.write(b"event: snapshot\ndata: " + json.dumps(dash.snapshot()).encode() + b"\n\n")
                self.wfile.flush()
                while True:
                    try:
                        entry = q.get(timeout=1.0)
                        payload = {"entry": entry, "snapshot": dash.snapshot()}
                        self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                dash.unsubscribe(q)

    return Handler


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Visual Actions</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaf0;--quiet:#8b93a7;--line:#262b36;--ok:#4cd27a;--warn:#f2b84b;--bad:#ef5a6f;--accent:#5aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,system-ui,sans-serif;padding:16px}
h1{font-size:16px;margin:0 0 12px;display:flex;gap:12px;align-items:center}h1 small{color:var(--quiet);font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.card h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--quiet);margin:0 0 8px}
.big{font-size:28px;font-weight:600}.mode-idle{color:var(--quiet)}.mode-holding{color:var(--accent)}.mode-armed{color:var(--ok)}.mode-dragging{color:#c48cff}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;background:var(--accent);width:0}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);font-size:13px}th{color:var(--quiet);font-weight:500}
.kv td:first-child{color:var(--quiet);width:46%}
.log{max-height:360px;overflow:auto;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.log div{padding:3px 0;border-bottom:1px solid var(--line)}.k-action{color:var(--ok)}.k-mode{color:var(--accent)}.k-drag{color:#c48cff}.k-token{color:var(--quiet)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--bad);margin-right:6px}.dot.on{background:var(--ok)}
</style></head><body>
<h1><span class="dot" id="dot"></span>Visual Actions <small id="up"></small></h1>
<div class="grid">
 <div class="card"><h2>Mode</h2><div class="big" id="mode">idle</div><div id="ns" style="color:var(--quiet)"></div><div class="bar"><i id="hold"></i></div></div>
 <div class="card"><h2>Last token</h2><div class="big" id="tok">—</div><div id="tokmeta" style="color:var(--quiet)"></div></div>
 <div class="card"><h2>Camera</h2><table class="kv"><tr><td>frames</td><td id="frames">—</td></tr><tr><td>past gate</td><td id="gate">—</td></tr><tr><td>capture error</td><td id="err">none</td></tr></table></div>
 <div class="card"><h2>Timing</h2><table class="kv" id="timing"></table></div>
 <div class="card"><h2>Drag</h2><table class="kv" id="drag"></table></div>
 <div class="card"><h2>Bindings</h2><table id="bind"><tr><th>namespace</th><th>gesture</th><th>action</th></tr></table></div>
 <div class="card" style="grid-column:1/-1"><h2>Events</h2><div class="log" id="log"></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
function kv(el,obj){el.innerHTML=Object.entries(obj).map(([k,v])=>`<tr><td>${k}</td><td>${typeof v==='number'?+v.toFixed(3):v}</td></tr>`).join('')}
function fmt(e){const t=new Date(e.t*1000).toLocaleTimeString();let s='';
 if(e.kind==='mode')s=`${e.old} → ${e.new}`;else if(e.kind==='token')s=`${e.name} ${e.conf}${e.still?' still':''}`;
 else if(e.kind==='action')s=`${e.name} ${e.ok?'ok':'FAILED '+e.message}`;else if(e.kind==='drag')s=`${e.phase} ${e.window}${e.snapped?' → snapped '+e.snapped:''}`;
 else if(e.kind==='hold')return '';return `<div class="k-${e.kind}">${t}  ${e.kind.padEnd(6)} ${s}</div>`}
function render(s){const st=s.state;$('mode').textContent=st.mode;$('mode').className='big mode-'+st.mode;$('ns').textContent=st.namespace?('namespace: '+st.namespace):(st.snap?('snap: '+st.snap):'');
 $('hold').style.width=(st.mode==='holding'?st.hold*100:(st.mode==='armed'||st.mode==='dragging')?100:0)+'%';
 $('tok').textContent=st.token||'—';$('tokmeta').textContent=st.token?`conf ${st.token_conf} · ${st.token_still?'still':'moving'} · ${st.tokens_per_s}/s`:'';
 const c=s.stats||{};$('frames').textContent=c.frames??'—';$('gate').textContent=c.frames?Math.round(100*(c.tracked||0)/c.frames)+'%':'—';$('err').textContent=c.error||'none';
 kv($('timing'),s.timing);kv($('drag'),s.drag);$('up').textContent='up '+s.uptime_s+'s · last action: '+(st.last_action||'—');
 $('bind').innerHTML='<tr><th>namespace</th><th>gesture</th><th>action</th></tr>'+s.bindings.map(b=>`<tr><td>${b.namespace}</td><td>${b.gesture}</td><td>${b.action}</td></tr>`).join('');
 $('log').innerHTML=s.log.map(fmt).join('')}
function connect(){const es=new EventSource('/events');es.addEventListener('snapshot',e=>{$('dot').className='dot on';render(JSON.parse(e.data))});
 es.onmessage=e=>render(JSON.parse(e.data).snapshot);es.onerror=()=>{$('dot').className='dot';es.close();setTimeout(connect,1500)}}
connect();setInterval(()=>fetch('/state.json').then(r=>r.json()).then(render).catch(()=>{}),2000);
</script></body></html>
"""
