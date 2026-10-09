"""Pass 2a: the JEV text state, questions, response parsing, combining, resume, retries."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from visual_actions.core import recognizer as R
from visual_actions.intent import jev
from visual_actions.intent.events import parse_events
from visual_actions.intent.schema import (
    CosmosVerdict,
    Intent,
    JevVerdict,
    Motion,
    Segment,
    SegmentKind,
    read_jsonl,
    write_jsonl,
)
from visual_actions.intent.segments import segment_session

ENGINE_NAMES = (R.NONE, R.OPEN_PALM, R.FIST, R.H_LEFT, R.H_RIGHT, R.POINT_UP, R.TWO_UP, R.MIDDLE_UP, R.THUMBS_UP, R.THUMBS_DOWN, "pinch")

LOG = """\
    0.000  23:12:52.309591  session started 2026-10-08 23:12:52 t0_ns=5000000000
    1.000  23:12:53.000000  hand    seen conf=0.99 wrist=(0.50,0.50) (mode idle)
    1.250  23:12:53.250000  token   open_palm conf=0.97 still=True
    1.250  23:12:53.250001  mode    idle -> holding [window]
    2.800  23:12:54.800000  mode    holding -> armed [window]
    3.000  23:12:55.000000  token   none conf=0.50 still=False
    3.300  23:12:55.300000  token   h_left conf=0.95 still=True
    3.300  23:12:55.300001  action  Cmd+Tab ok
    3.300  23:12:55.300002  mode    armed -> armed [window]
    3.800  23:12:55.800000  token   fist conf=0.90 still=True
    3.800  23:12:55.800001  mode    armed -> idle
    4.200  23:12:56.200000  veto    hand near face
   12.000  23:13:04.000000  hand    lost [tracker] tracker reported no hand (frame 300) (mode idle)
   25.000  23:13:17.000000  hand    seen after 13.00s away conf=0.95 wrist=(0.40,0.60) (mode idle)
   26.000  23:13:18.000000  hand    lost [edge] wrist outside (mode idle)
"""


def _events(tmp_path: Path):
    p = tmp_path / "events.log"
    p.write_text(LOG)
    return parse_events(p)


def _fire(tmp_path: Path) -> tuple[Segment, list]:
    ev = _events(tmp_path)
    segs = segment_session("s1", ev)
    return next(s for s in segs if s.kind is SegmentKind.FIRE), ev


def _cosmos(sid: str) -> CosmosVerdict:
    return CosmosVerdict(sid, True, True, "yes", True, False, Intent.COMMAND, Motion.STILL, "index finger extended, pointing to the viewer's left, held at shoulder height", "the hand is raised as a signal", 0.8)


def test_trace_and_state_carry_no_engine_vocabulary(tmp_path: Path):
    seg, ev = _fire(tmp_path)
    assert seg.engine_gesture == "h_left" and seg.engine_outcome == "cancelled_by_fist" and seg.weak_label == "misfire"
    trace = jev.build_trace(seg, ev)
    state = jev.render_state(seg, _cosmos(seg.segment_id), trace, jev.acted_at(seg, ev))
    low = state.lower()
    for name in ENGINE_NAMES:
        assert name not in low, name
    for leak in ("cancelled", "misfire", "intended", "weak", "fire", "armed", "cmd+tab", "window"):
        assert leak not in low, leak
    # shape ids by order of appearance, the no-shape frame as '-', timings relative to t0
    assert trace.startswith("+1.20s - c=0.50 moving; +1.50s S1 c=0.95 still") and "+2.00s S2 c=0.90 still" in trace
    assert "+1.50s system acted" in trace and "hand near the face" in trace
    assert "the system acted at +1.5 s" in state and "hand in view: 100% of the clip" in state
    assert "hand, as seen: index finger extended" in state and "hand motion: still" in state


def test_state_without_cosmos_or_action():
    seg = Segment("s/hand/0000", "s", SegmentKind.HAND, 10.0, 16.0, hand_present_fraction=0.42)
    state = jev.render_state(seg, None, "")
    assert "video description: not available" in state and "the system did not act" in state and "42%" in state and "no shapes detected" in state
    assert len(jev.render_state(seg, None, "x" * 20000)) <= jev.MAX_STATE_CHARS


def test_question_ids_are_stable_and_options_are_physical():
    assert jev.QUESTION_IDS == ("deliberate_sign", "unrelated_activity", "held_still", "sideways_stroke", "reacts_unwanted", "repeated_sign", "nobody_at_desk", "hand_shape")
    q = jev.questions()
    assert set(q) == set(jev.QUESTION_IDS)
    assert all(q[i]["type"] == "noul" and q[i]["instructions"] for i in jev.NOUL_QUESTIONS)
    shape = q[jev.Q_SHAPE]
    assert shape["type"] == "choice" and "none" in shape["criteria"]
    for opt in shape["criteria"]:
        assert opt == "none" or opt not in ENGINE_NAMES, opt
    assert set(shape["criteria"]) == set(jev.OPTION_TO_ENGINE)
    assert {v for v in jev.OPTION_TO_ENGINE.values() if v} == set(ENGINE_NAMES) - {"none"}
    q[jev.Q_SHAPE]["criteria"].clear()
    assert jev.questions()[jev.Q_SHAPE]["criteria"]  # a fresh copy each time


def _response(**nouls: float) -> dict[str, Any]:
    answers: dict[str, Any] = {k: {"type": "noul", "noul": v} for k, v in nouls.items()}
    answers[jev.Q_SHAPE] = {"type": "choice", "choice": "index_left", "confidence": 0.7, "probabilities": {"index_left": 0.73, "none": 0.2, "spread_palm": 0.07}}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 8}}


def test_parse_response_and_engine_mapping():
    v = jev.parse_response("s/fire/0000", _response(deliberate_sign=0.9, unrelated_activity=0.1, held_still=0.8, nobody_at_desk=0.02))
    assert v.answers == {"deliberate_sign": 0.9, "unrelated_activity": 0.1, "held_still": 0.8, "nobody_at_desk": 0.02}
    assert v.choice == "index_left" and v.choice_probs["index_left"] == 0.73 and v.model == "jev-1.13.0"
    assert jev.choice_to_engine(v.choice) == "h_left" and jev.choice_to_engine("none") is None and jev.choice_to_engine(None) is None
    assert 0.7 < v.confidence < 0.9
    # partial / malformed answers are skipped, not fatal; choice confidence falls back to the docs' formula
    v2 = jev.parse_response("x", {"answers": {"deliberate_sign": {"type": "noul", "noul": "high"}, "hand_shape": {"choice": "none", "probabilities": {"none": 1.0, "spread_palm": 0.0}}}})
    assert v2.answers == {} and v2.choice == "none" and v2.confidence == 1.0
    assert jev.parse_response("x", {}).confidence == 0.0


def test_deliberate_sign_score_combines_atomic_answers():
    clear = {"deliberate_sign": 0.95, "unrelated_activity": 0.05, "held_still": 0.9, "nobody_at_desk": 0.0}
    typing = {"deliberate_sign": 0.6, "unrelated_activity": 0.9, "held_still": 0.2, "nobody_at_desk": 0.0}
    empty = {"deliberate_sign": 0.3, "unrelated_activity": 0.1, "held_still": 0.1, "nobody_at_desk": 0.95}
    stroke = {"deliberate_sign": 0.9, "unrelated_activity": 0.1, "held_still": 0.1, "sideways_stroke": 0.9}
    assert jev.deliberate_sign_score(clear) > 0.8
    assert jev.deliberate_sign_score(typing) < 0.1
    assert jev.deliberate_sign_score(empty) < 0.05
    assert jev.deliberate_sign_score(stroke) > 0.7  # a stroke counts as evidence like a held shape
    assert jev.deliberate_sign_score({}) == 0.25 and jev.noul_confidence(0.5) == 0.0 and jev.noul_confidence(0.95) == pytest.approx(0.9)


def test_run_jev_resumes_and_honours_limit(tmp_path: Path):
    ev = _events(tmp_path)
    segs = segment_session("s1", ev)
    d = tmp_path / "intent"
    write_jsonl(d / "segments.jsonl", segs)
    write_jsonl(d / "cosmos.jsonl", [_cosmos(segs[0].segment_id)])
    fake = jev.FakeJev(lambda state, q: _response(deliberate_sign=0.9, unrelated_activity=0.1, held_still=0.8, nobody_at_desk=0.0))
    logs: list[str] = []
    assert jev.run_jev(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", fake, limit=2, log=logs.append) == 2
    assert len(fake.requests) == 2 and len(logs) == 2
    state, questions = fake.requests[0]
    assert set(questions) == set(jev.QUESTION_IDS) and "hand, as seen" in state  # cosmos joined by id
    n_rest = jev.run_jev(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", fake, log=logs.append)
    assert n_rest == len(segs) - 2
    assert jev.run_jev(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", fake, log=logs.append) == 0
    out = list(read_jsonl(d / "jev.jsonl", JevVerdict))
    assert [v.segment_id for v in out] == [s.segment_id for s in segs] and out[0].model == "jev-1.13.0"


class _Resp:
    def __init__(self, body: dict[str, Any]):
        self._b = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._b


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", code, "err", None, io.BytesIO(b'{"error":"x"}'))  # type: ignore[arg-type]


def test_client_request_shape_and_retry_on_429_then_529():
    seen: list[Any] = []
    codes = [429, 529]

    def urlopen(req, timeout):
        seen.append(req)
        if codes:
            raise _http_error(codes.pop(0))
        return _Resp(_response(deliberate_sign=0.8))

    slept: list[float] = []
    c = jev.JevClient(api_key="k", sleep=slept.append, urlopen=urlopen, backoff_s=0.5)
    resp = c.request("state", jev.questions())
    assert resp["model"] == "jev-1.13.0" and c.calls == 3 and slept == [0.5, 1.0]
    req = seen[0]
    assert req.full_url == jev.JEV_URL and req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer k" and req.get_header("Content-type") == "application/json"
    body = json.loads(req.data)
    assert body["state"] == "state" and body["model"] == "jev-latest" and body["questions"][jev.Q_SHAPE]["type"] == "choice"


def test_client_does_not_retry_401_or_422_and_gives_up_after_max():
    def bad(req, timeout):
        raise _http_error(422)

    c = jev.JevClient(api_key="k", sleep=lambda s: None, urlopen=bad)
    with pytest.raises(jev.JevError) as ei:
        c.request("s", jev.questions())
    assert ei.value.status == 422 and c.calls == 1

    def down(req, timeout):
        raise _http_error(503)

    c = jev.JevClient(api_key="k", sleep=lambda s: None, urlopen=down, max_retries=2)
    with pytest.raises(jev.JevError) as ei:
        c.request("s", jev.questions())
    assert ei.value.status == 503 and c.calls == 3
    with pytest.raises(jev.JevError) as ei:
        jev.JevClient(api_key="", urlopen=down).request("s", {})
    assert ei.value.status == 401
