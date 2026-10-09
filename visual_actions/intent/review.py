"""The human review queue: which moments a person should look at, and a one-page server to
label them one key at a time.

The queue joins segments.jsonl with cosmos / jev / tags by segment_id. A moment is queued when
the tagger asked for a human, when no tag exists and the weak label is doubtful, or when an
engine event (arm, fire) has not been judged at all. A small deterministic audit sample of the
confident tags is queued too, so the judges are checked. Already-labelled ids are skipped, the
queue is ordered by how much a label is worth, and capped.

With decisions=True the same page is a read-only dashboard of what the AI decided: every
tagged segment (plus untagged fire / arm segments, marked "not judged yet") in time order,
uncapped, with the tag's verdict and reason, a summary of counts by verdict and how many were
escalated to the stronger model; labels and undo answer 403 there.

Each answer is one key and one appended line in <session>/intent/human.jsonl. Standard library
only, like ui/dashboard.py; clips come from intent/clips.py when it is importable and the page
falls back to a skeleton strip, then to the text panel alone.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .schema import (
    CosmosVerdict,
    HumanLabel,
    Intent,
    JevVerdict,
    Motion,
    Segment,
    SegmentKind,
    Tag,
    Verdict,
    append_jsonl,
    by_id,
    read_jsonl,
)

QUEUE_CAP = 300
AUDIT_FRACTION = 0.1  # of confident tags that did not ask for a human
HIGH_CONF = 0.8
DOUBTFUL_WEAK = ("misfire", "unsure")

# Priority buckets, lowest first in the queue.
DISAGREEMENT, HIGH_MISFIRE, AUDIT, REST, DECISION = 0, 1, 2, 3, 4
BUCKET_NAMES = {DISAGREEMENT: "judge vs labelfn disagree", HIGH_MISFIRE: "confident misfire", AUDIT: "audit sample", REST: "needs a look", DECISION: "decision"}
ESCALATED_MARK = "[escalated"
NOT_JUDGED = "not judged yet"

# Key -> engine gesture, shown as the legend on the page.
GESTURE_KEYS: dict[str, str | None] = {
    "p": "open_palm",
    "f": "fist",
    "l": "h_left",
    "r": "h_right",
    "1": "point_up",
    "2": "two_up",
    "3": "middle_up",
    "u": "thumbs_up",
    "d": "thumbs_down",
    "i": "pinch",
    "0": None,
}
VERDICT_KEYS: dict[str, Verdict] = {"y": Verdict.INTENDED, "n": Verdict.MISFIRE, "m": Verdict.MISSED, "x": Verdict.NO_EVENT, "a": Verdict.AMBIGUOUS}


@dataclass
class ReviewItem:
    segment: Segment
    cosmos: CosmosVerdict | None = None
    jev: JevVerdict | None = None
    tag: Tag | None = None
    bucket: int = REST
    reason: str = ""
    value: float = 0.0  # within a bucket, higher is reviewed first

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id

    @property
    def bucket_name(self) -> str:
        return BUCKET_NAMES.get(self.bucket, "")

    def to_json(self) -> dict[str, Any]:
        return {
            "segment": asdict(self.segment),
            "cosmos": asdict(self.cosmos) if self.cosmos else None,
            "jev": asdict(self.jev) if self.jev else None,
            "tag": asdict(self.tag) if self.tag else None,
            "bucket": self.bucket,
            "bucket_name": self.bucket_name,
            "reason": self.reason,
            "value": round(self.value, 3),
        }


def clip_name(segment_id: str) -> str:
    """A segment id is "<session>/<kind>/<index>"; the clip file flattens it."""
    return segment_id.replace("/", "_")


def intent_dir(session_dir: Path) -> Path:
    return session_dir / "intent"


def human_path(session_dir: Path) -> Path:
    return intent_dir(session_dir) / "human.jsonl"


def load_passes(session_dir: Path) -> tuple[list[Segment], dict[str, CosmosVerdict], dict[str, JevVerdict], dict[str, Tag], dict[str, HumanLabel]]:
    d = intent_dir(session_dir)
    segments = list(read_jsonl(d / "segments.jsonl", Segment))
    cosmos = by_id(read_jsonl(d / "cosmos.jsonl", CosmosVerdict))
    jev = by_id(read_jsonl(d / "jev.jsonl", JevVerdict))
    tags = by_id(read_jsonl(d / "tags.jsonl", Tag))
    human = by_id(read_jsonl(d / "human.jsonl", HumanLabel))
    return segments, cosmos, jev, tags, human


def _in_audit_sample(segment_id: str, fraction: float) -> bool:
    h = int(hashlib.sha1(segment_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def classify(seg: Segment, tag: Tag | None, audit_fraction: float = AUDIT_FRACTION) -> tuple[int, str] | None:
    """(bucket, reason) when the segment belongs in the queue, else None."""
    weak = seg.weak_label
    if tag is not None:
        if tag.needs_human:
            if weak in ("intended", "misfire", "missed") and tag.verdict.value != weak:
                return DISAGREEMENT, f"judge says {tag.verdict.value}, labelfns say {weak}"
            if tag.verdict is Verdict.MISFIRE and tag.confidence >= HIGH_CONF:
                return HIGH_MISFIRE, f"judge: misfire at {tag.confidence:.2f}"
            return REST, f"judge asked: {tag.reason or 'needs a human'}"
        if weak in ("intended", "misfire", "missed") and tag.verdict.value != weak:
            return DISAGREEMENT, f"judge says {tag.verdict.value} ({tag.confidence:.2f}), labelfns say {weak}"
        if _in_audit_sample(seg.segment_id, audit_fraction):
            return AUDIT, f"audit of a confident {tag.verdict.value}"
        return None
    # no tag: the judges have not run (or skipped this one)
    if weak == "misfire" and seg.weak_weight >= 0.5:
        return HIGH_MISFIRE, f"labelfns: misfire at {seg.weak_weight:.2f}"
    if weak in DOUBTFUL_WEAK:
        return REST, f"labelfns: {weak}, no judge"
    if seg.kind in (SegmentKind.FIRE, SegmentKind.ARM):
        return REST, f"{seg.kind.value} without a judge"
    return None


def _value(item: ReviewItem) -> float:
    seg, tag = item.segment, item.tag
    if tag is not None:
        return tag.confidence
    return seg.weak_weight if seg.weak_label else (seg.mean_confidence or 0.0)


def build_queue(session_dir: Path, cap: int = QUEUE_CAP, audit_fraction: float = AUDIT_FRACTION, decisions: bool = False) -> list[ReviewItem]:
    """The human queue; with `decisions` the dashboard list: every tagged segment plus the
    untagged fire / arm ones, ordered by t0, uncapped, bucket DECISION."""
    segments, cosmos, jev, tags, human = load_passes(session_dir)
    items: list[ReviewItem] = []
    if decisions:
        for seg in segments:
            tag = tags.get(seg.segment_id)
            if tag is not None:
                reason = f"{tag.verdict.value} ({tag.confidence:.2f}): {tag.reason}"
            elif seg.kind in (SegmentKind.FIRE, SegmentKind.ARM):
                reason = NOT_JUDGED
            else:
                continue
            item = ReviewItem(seg, cosmos.get(seg.segment_id), jev.get(seg.segment_id), tag, DECISION, reason)
            item.value = _value(item)
            items.append(item)
        items.sort(key=lambda it: it.segment.t0)
        return items
    for seg in segments:
        if seg.segment_id in human:
            continue
        tag = tags.get(seg.segment_id)
        c = classify(seg, tag, audit_fraction)
        if c is None:
            continue
        item = ReviewItem(seg, cosmos.get(seg.segment_id), jev.get(seg.segment_id), tag, c[0], c[1])
        item.value = _value(item)
        items.append(item)
    items.sort(key=lambda it: (it.bucket, -it.value, it.segment.t0))
    return items[:cap]


def summarize(items: list[ReviewItem]) -> dict[str, Any]:
    """Counts by verdict, how many were escalated to the stronger model, how many are not
    judged yet, and which models answered."""
    by_verdict: dict[str, int] = {}
    models: dict[str, int] = {}
    escalated = 0
    unjudged = 0
    for it in items:
        if it.tag is None:
            unjudged += 1
            continue
        by_verdict[it.tag.verdict.value] = by_verdict.get(it.tag.verdict.value, 0) + 1
        models[it.tag.model or "?"] = models.get(it.tag.model or "?", 0) + 1
        if ESCALATED_MARK in it.tag.reason:
            escalated += 1
    return {"total": len(items), "by_verdict": by_verdict, "escalated": escalated, "not_judged": unjudged, "models": models}


# -- the server --------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _default_intent(seg: Segment, verdict: Verdict) -> Intent:
    if verdict in (Verdict.INTENDED, Verdict.MISSED):
        return Intent.COMMAND
    if verdict is Verdict.AMBIGUOUS:
        return Intent.UNSURE
    if seg.kind is SegmentKind.DEAD or (seg.hand_present_fraction is not None and seg.hand_present_fraction == 0.0):
        return Intent.DEAD
    return Intent.INCIDENTAL


class ReviewServer:
    """Serves the queue of one session. `label` and `undo` are the two writes; the HTTP side
    is a thin JSON wrapper over them so the page and the tests share one path."""

    def __init__(self, session_dir: Path, port: int = 8766, host: str = "127.0.0.1", cap: int = QUEUE_CAP, readonly: bool = False) -> None:
        self.session_dir = Path(session_dir)
        self.readonly = readonly
        self.queue = build_queue(self.session_dir, cap=cap, decisions=readonly)
        self.summary = summarize(self.queue) if readonly else {}
        self.pos = 0
        self.history: list[tuple[int, HumanLabel]] = []  # (queue index, label) for undo
        self.done = 0
        self._lock = threading.RLock()
        self.server = ThreadingHTTPServer((host, port), _handler_for(self))
        self.port = self.server.server_address[1]
        self.url = f"http://{host}:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, name="review-http", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- state ----------------------------------------------------------------------

    def current(self) -> dict[str, Any]:
        with self._lock:
            total = len(self.queue)
            item = self.queue[self.pos] if 0 <= self.pos < total else None
            last = self.history[-1][1] if self.history else None
            return {
                "readonly": self.readonly,
                "summary": self.summary,
                "title": "Decisions (read-only)" if self.readonly else "Intent review",
                "index": self.pos,
                "total": total,
                "done": self.done,
                "remaining": max(0, total - self.pos),
                "item": item.to_json() if item else None,
                "media": self._media(item) if item else {},
                "last": asdict(last) if last else None,
                "legend": {k: (v or "none") for k, v in GESTURE_KEYS.items()},
                "order": [{"segment_id": it.segment_id, "bucket": it.bucket_name, "reason": it.reason} for it in self.queue[self.pos : self.pos + 8]],
            }

    def _media(self, item: ReviewItem) -> dict[str, Any]:
        name = clip_name(item.segment_id)
        clips = _clips_module()
        clip_file = intent_dir(self.session_dir) / "clips" / f"{name}.mp4"
        return {
            "clip": f"/clip/{name}.mp4" if clip_file.exists() or (clips is not None and hasattr(clips, "clip_mp4")) else None,
            "strip": f"/strip/{name}.png" if clips is not None and hasattr(clips, "skeleton_strip") else None,
        }

    def label(self, segment_id: str, verdict: str, true_gesture: str | None = None, intent: str | None = None, motion: str | None = None, note: str = "", amend: bool = False) -> dict[str, Any]:
        if self.readonly:
            raise PermissionError("read-only: the decisions dashboard takes no labels")
        with self._lock:
            idx = next((i for i, it in enumerate(self.queue) if it.segment_id == segment_id), None)
            if idx is None:
                raise KeyError(segment_id)
            seg = self.queue[idx].segment
            v = Verdict(verdict)
            if amend and self.history and self.history[-1][1].segment_id == segment_id:
                prev = self.history.pop()[1]
                _drop_last(human_path(self.session_dir), segment_id)
                self.done -= 1
                intent = intent or prev.intent.value
                motion = motion or (prev.motion.value if prev.motion else None)
                note = note or prev.note
            if true_gesture is None and v is Verdict.INTENDED:
                true_gesture = seg.engine_gesture
            rec = HumanLabel(
                segment_id=segment_id,
                verdict=v,
                intent=Intent(intent) if intent else _default_intent(seg, v),
                true_gesture=true_gesture,
                motion=Motion(motion) if motion else None,
                note=note,
                at=_now_iso(),
            )
            append_jsonl(human_path(self.session_dir), rec)
            self.history.append((idx, rec))
            self.done += 1
            if idx == self.pos:
                self.pos += 1
        return self.current()

    def undo(self) -> dict[str, Any]:
        if self.readonly:
            raise PermissionError("read-only: nothing to undo")
        with self._lock:
            if not self.history:
                return self.current()
            idx, rec = self.history.pop()
            _drop_last(human_path(self.session_dir), rec.segment_id)
            self.done -= 1
            self.pos = idx
        return self.current()

    def skip(self) -> dict[str, Any]:
        with self._lock:
            if self.pos < len(self.queue):
                self.pos += 1
        return self.current()

    def goto(self, index: int) -> dict[str, Any]:
        with self._lock:
            self.pos = max(0, min(index, len(self.queue)))
        return self.current()

    # -- media ------------------------------------------------------------------------

    def clip_bytes(self, name: str) -> bytes | None:
        item = self._by_name(name)
        if item is None:
            return None
        out = intent_dir(self.session_dir) / "clips" / f"{name}.mp4"
        if not out.exists():
            clips = _clips_module()
            if clips is None or not hasattr(clips, "clip_mp4"):
                return None
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                clips.clip_mp4(self.session_dir, item.segment.t0, item.segment.t1, out)
            except Exception:
                return None
        return out.read_bytes() if out.exists() else None

    def strip_bytes(self, name: str) -> bytes | None:
        item = self._by_name(name)
        clips = _clips_module()
        if item is None or clips is None or not hasattr(clips, "skeleton_strip"):
            return None
        try:
            strip = clips.skeleton_strip(self.session_dir, item.segment.t0, item.segment.t1)
        except Exception:
            return None
        return _as_png(strip)

    def _by_name(self, name: str) -> ReviewItem | None:
        return next((it for it in self.queue if clip_name(it.segment_id) == name), None)


def _clips_module() -> Any | None:
    try:
        from . import clips  # written by another pass; optional at runtime
    except ImportError:
        return None
    return clips


def _as_png(strip: Any) -> bytes | None:
    """Whatever skeleton_strip returns: PNG bytes, a path to one, or an image array."""
    if strip is None:
        return None
    if isinstance(strip, (bytes, bytearray)):
        return bytes(strip)
    if isinstance(strip, (str, Path)):
        p = Path(strip)
        return p.read_bytes() if p.exists() else None
    try:
        import cv2  # type: ignore[import-not-found]

        ok, buf = cv2.imencode(".png", strip)
        return buf.tobytes() if ok else None
    except Exception:
        return None


def _drop_last(path: Path, segment_id: str) -> None:
    """Remove the last line of human.jsonl for `segment_id` (undo). The file is small."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() and json.loads(lines[i]).get("segment_id") == segment_id:
            del lines[i]
            break
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _handler_for(srv: ReviewServer) -> type[BaseHTTPRequestHandler]:
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

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                d = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                d = {}
            return d if isinstance(d, dict) else {}

        def do_GET(self) -> None:
            u = urlsplit(self.path)
            if u.path == "/":
                title = "Decisions (read-only)" if srv.readonly else "Intent review"
                self._send(200, PAGE.replace("Intent review", title).encode(), "text/html; charset=utf-8")
            elif u.path == "/item":
                q = parse_qs(u.query)
                if "i" in q:
                    self._json(srv.goto(int(q["i"][0])))
                else:
                    self._json(srv.current())
            elif u.path.startswith("/clip/") and u.path.endswith(".mp4"):
                data = srv.clip_bytes(u.path[len("/clip/") : -4])
                self._send(200, data, "video/mp4") if data else self._send(404, b"no clip", "text/plain")
            elif u.path.startswith("/strip/") and u.path.endswith(".png"):
                data = srv.strip_bytes(u.path[len("/strip/") : -4])
                self._send(200, data, "image/png") if data else self._send(404, b"no strip", "text/plain")
            elif u.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            u = urlsplit(self.path)
            try:
                if u.path == "/label":
                    b = self._body()
                    self._json(srv.label(b["segment_id"], b["verdict"], b.get("true_gesture"), b.get("intent"), b.get("motion"), b.get("note", ""), bool(b.get("amend"))))
                elif u.path == "/undo":
                    self._json(srv.undo())
                elif u.path == "/skip":
                    self._json(srv.skip())
                else:
                    self._send(404, b"not found", "text/plain")
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except (KeyError, ValueError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    return Handler


def serve(session_dir: Path, port: int = 8766, decisions: bool = False) -> int:
    srv = ReviewServer(session_dir, port=port, readonly=decisions)
    if decisions:
        sm = srv.summary
        print(f"decisions {session_dir.name}: {sm['total']} moments, " + " ".join(f"{k}={v}" for k, v in sorted(sm["by_verdict"].items())) + f" escalated={sm['escalated']} not_judged={sm['not_judged']} at {srv.url} (read-only; Ctrl-C to stop)")
    else:
        print(f"review {session_dir.name}: {len(srv.queue)} in the queue at {srv.url}  (q in the page or Ctrl-C here to stop)")
    try:
        srv._thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
    return 0


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Intent review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaf0;--quiet:#8b93a7;--line:#262b36;--ok:#4cd27a;--warn:#f2b84b;--bad:#ef5a6f;--accent:#5aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,system-ui,sans-serif;padding:16px}
h1{font-size:16px;margin:0 0 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1 small{color:var(--quiet);font-weight:400}
.grid{display:grid;grid-template-columns:minmax(320px,1.4fr) minmax(280px,1fr);gap:12px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.card h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--quiet);margin:0 0 8px}
video,img{width:100%;border-radius:8px;background:#000;display:block}img.strip{margin-top:8px;background:transparent}
table{width:100%;border-collapse:collapse}td{text-align:left;padding:3px 6px;border-bottom:1px solid var(--line);font-size:13px;vertical-align:top}
.kv td:first-child{color:var(--quiet);width:38%}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap}
.keys{display:flex;flex-wrap:wrap;gap:6px}.keys span{border:1px solid var(--line);border-radius:6px;padding:2px 8px;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.keys b{color:var(--accent)}.v-intended{color:var(--ok)}.v-misfire{color:var(--bad)}.v-missed{color:var(--warn)}.v-no_event{color:var(--quiet)}.v-ambiguous{color:#c48cff}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:6px 0}.bar i{display:block;height:100%;background:var(--accent)}
.toast{position:fixed;right:16px;bottom:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;opacity:0;transition:opacity .2s}.toast.on{opacity:1}
.queue div{color:var(--quiet);font-size:12px;padding:2px 0}.queue div:first-child{color:var(--ink)}
.reason{color:var(--warn)}
</style></head><body>
<h1>Intent review <small id="prog"></small> <small id="reason" class="reason"></small></h1>
<div id="summary" style="color:var(--quiet);margin:-6px 0 8px;display:none"></div>
<div class="bar"><i id="bar"></i></div>
<div class="grid">
 <div>
  <div class="card" id="media"><h2>Moment</h2><div id="mediabox"></div></div>
  <div class="card" style="margin-top:12px"><h2>Keys</h2><div class="keys">
   <span><b>y</b> intended</span><span><b>n</b> misfire</span><span><b>m</b> missed</span><span><b>x</b> no event</span><span><b>a</b> ambiguous</span>
   <span><b>g</b>+letter true gesture of the last answer</span><span><b>space</b> skip</span><span><b>u</b> undo</span><span><b>q</b> quit</span></div>
   <div class="keys" id="legend" style="margin-top:8px"></div>
   <div id="last" style="margin-top:8px;color:var(--quiet)"></div></div>
  <div class="card queue" style="margin-top:12px"><h2>Up next (why)</h2><div id="queue"></div></div>
 </div>
 <div>
  <div class="card"><h2>Engine</h2><table class="kv" id="engine"></table></div>
  <div class="card" style="margin-top:12px"><h2>Labelling functions</h2><table class="kv" id="lf"></table></div>
  <div class="card" style="margin-top:12px"><h2>Cosmos saw</h2><div id="cosmos" class="mono"></div></div>
  <div class="card" style="margin-top:12px"><h2>JEV</h2><table class="kv" id="jev"></table></div>
  <div class="card" style="margin-top:12px"><h2>Tag</h2><table class="kv" id="tag"></table></div>
 </div>
</div>
<div class="toast" id="toast"></div>
<script>
const $=id=>document.getElementById(id);let S=null,gmode=false,quit=false;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function kv(el,obj){el.innerHTML=Object.entries(obj||{}).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(typeof v==='number'?+v.toFixed(3):(v===null?'—':(typeof v==='object'?JSON.stringify(v):v)))}</td></tr>`).join('')||'<tr><td colspan=2>—</td></tr>'}
function toast(t){const e=$('toast');e.textContent=t;e.className='toast on';clearTimeout(e._t);e._t=setTimeout(()=>e.className='toast',1200)}
function summary(s){const e=$('summary');if(!s.readonly||!s.summary){e.style.display='none';return}const m=s.summary;
 e.style.display='block';e.innerHTML=`<b>${m.total}</b> decisions · `+Object.entries(m.by_verdict||{}).map(([k,v])=>`<span class="v-${k}">${v} ${k}</span>`).join(' · ')+` · <b>${m.escalated}</b> escalated to the stronger model · ${m.not_judged} not judged yet · models: ${esc(Object.entries(m.models||{}).map(([k,v])=>k+' ×'+v).join(', ')||'—')}`}
function render(s){S=s;summary(s);const it=s.item;$('prog').textContent=`${s.done} labelled · ${s.index+1 > s.total ? s.total : s.index+1} of ${s.total}`;
 $('bar').style.width=(s.total?100*s.index/s.total:0)+'%';
 $('legend').innerHTML=Object.entries(s.legend).map(([k,v])=>`<span><b>g ${k}</b> ${v}</span>`).join('');
 $('last').textContent=s.last?`last: ${s.last.segment_id} → ${s.last.verdict}${s.last.true_gesture?' · '+s.last.true_gesture:''} (${s.last.intent})`:'';
 $('queue').innerHTML=(s.order||[]).map(o=>`<div>${esc(o.segment_id)} — ${esc(o.bucket)}: ${esc(o.reason)}</div>`).join('');
 if(!it){$('reason').textContent=s.total?'queue done':'nothing to review';$('mediabox').innerHTML='<div style="color:var(--quiet)">Nothing left in the queue.</div>';
  ['engine','lf','jev','tag'].forEach(i=>kv($(i),{}));$('cosmos').textContent='';return}
 const sg=it.segment;$('reason').textContent=`${it.bucket_name}: ${it.reason}`;
 let m='';if(s.media.clip)m+=`<video src="${s.media.clip}" autoplay loop muted playsinline onerror="this.style.display='none'"></video>`;
 if(s.media.strip)m+=`<img class="strip" src="${s.media.strip}" onerror="this.style.display='none'">`;
 m+=`<div class="mono" style="margin-top:8px">${esc(sg.segment_id)}  ${sg.kind}  ${sg.t0.toFixed(2)}–${sg.t1.toFixed(2)} s${sg.notes?'\\n'+esc(sg.notes):''}</div>`;$('mediabox').innerHTML=m;
 kv($('engine'),{gesture:sg.engine_gesture,action:sg.engine_action,outcome:sg.engine_outcome,namespace:sg.engine_namespace,confidence:sg.mean_confidence,'hand present':sg.hand_present_fraction});
 const lf=Object.assign({},sg.labelfn_votes);lf['= weak label']=sg.weak_label?`${sg.weak_label} (${sg.weak_weight.toFixed(2)})`:'—';kv($('lf'),lf);
 const c=it.cosmos;$('cosmos').textContent=c?`${c.intent} · ${c.motion} · conf ${c.confidence}\\nperson ${c.person_present} hand ${c.hand_present} attention ${c.attention_to_screen} arm ${c.arm_raised_toward_camera} face ${c.face_touched}\\n${c.hand_description}\\n${c.reasoning}`:'—';
 const j=it.jev;kv($('jev'),j?Object.assign({choice:j.choice,confidence:j.confidence},j.answers,Object.fromEntries(Object.entries(j.choice_probs||{}).map(([k,v])=>['p('+k+')',v]))):{});
 const t=it.tag;kv($('tag'),t?{verdict:t.verdict,intent:t.intent,motion:t.motion,'true gesture':t.true_gesture,tags:(t.tags||[]).join(', '),confidence:t.confidence,'needs human':t.needs_human,reason:t.reason}:{});}
async function get(){render(await (await fetch('/item')).json())}
async function post(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});const j=await r.json();if(j.error)toast(j.error);else render(j)}
const V={y:'intended',n:'misfire',m:'missed',x:'no_event',a:'ambiguous'};
document.addEventListener('keydown',async e=>{if(quit||e.metaKey||e.ctrlKey||e.altKey)return;const k=e.key;
 if(S&&S.readonly){if(k===' '||k==='ArrowRight'){e.preventDefault();await post('/skip')}else if(k==='ArrowLeft'){e.preventDefault();render(await (await fetch('/item?i='+Math.max(0,S.index-1))).json())}else if(k==='q'){quit=true;$('reason').textContent='quit — close the tab, Ctrl-C the server'}else if(k in V||k==='g'||k==='u')toast('read-only: decisions dashboard');return}
 if(gmode){gmode=false;if(S&&S.last&&(k in S.legend)){await post('/label',{segment_id:S.last.segment_id,verdict:S.last.verdict,true_gesture:S.legend[k]==='none'?null:S.legend[k],amend:true});toast('gesture: '+S.legend[k])}else toast('no gesture');return}
 if(k in V){if(S&&S.item){e.preventDefault();await post('/label',{segment_id:S.item.segment.segment_id,verdict:V[k]});toast(V[k])}}
 else if(k==='g'){gmode=true;toast('g: pick a gesture for the last answer')}
 else if(k===' '){e.preventDefault();await post('/skip');toast('skipped')}
 else if(k==='u'){await post('/undo');toast('undone')}
 else if(k==='q'){quit=true;$('reason').textContent='quit — close the tab, Ctrl-C the server';}
});
get();
</script></body></html>
"""
