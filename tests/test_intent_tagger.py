"""Pass 2b: the tagger prompt, lenient JSON parsing, the needs_human rule, escalation, retries, resume."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from visual_actions.intent import tagger
from visual_actions.intent.schema import (
    CosmosVerdict,
    Intent,
    JevVerdict,
    Motion,
    Segment,
    SegmentKind,
    Tag,
    Verdict,
    read_jsonl,
    write_jsonl,
)
from visual_actions.intent.tagger import Answer

SID = "s1/fire/0000"


def _seg(**kw: Any) -> Segment:
    base: dict[str, Any] = {"engine_gesture": "h_left", "engine_action": "Cmd+Tab", "engine_outcome": "cancelled_by_fist", "engine_namespace": "window", "mean_confidence": 0.95, "hand_present_fraction": 1.0, "labelfn_votes": {"fist_after_fire": "misfire"}, "weak_label": "misfire", "weak_weight": 0.6}
    base.update(kw)
    return Segment(SID, "s1", SegmentKind.FIRE, 1.8, 4.8, **base)


def _cosmos(intent: Intent = Intent.COMMAND) -> CosmosVerdict:
    return CosmosVerdict(SID, True, True, "yes", True, False, intent, Motion.STILL, "index finger out, pointing left", "raised as a signal", 0.8)


def _jev(sign: float = 0.9) -> JevVerdict:
    return JevVerdict(SID, {"deliberate_sign": sign, "unrelated_activity": 0.05, "held_still": 0.9, "nobody_at_desk": 0.0}, "index_left", {"index_left": 0.8, "none": 0.2}, 0.8, "jev-1.13.0")


def _answer(verdict: Verdict = Verdict.MISFIRE, conf: float = 0.9) -> Answer:
    return Answer(verdict, Intent.COMMAND, Motion.STILL, "h_left", ["held_from_previous_command"], conf, "why")


def test_prompt_carries_engine_video_decision_and_votes():
    p = tagger.build_prompt(_seg(), _cosmos(), _jev(), {"fist_after_fire": "misfire", "clean_command": "intended"})
    assert "## Engine" in p and "token=h_left action=Cmd+Tab outcome=cancelled_by_fist" in p
    assert '"clean_command": "intended"' in p and "weak_label=misfire weight=0.60" in p
    assert "## Video model" in p and "hand: index finger out, pointing left" in p and "intent=command" in p
    assert "## Decision model" in p and "deliberate_sign: 0.90" in p and "combined deliberate-sign score:" in p and "hand shape pick: index_left (engine: h_left)" in p
    p2 = tagger.build_prompt(_seg(), None, None)
    assert p2.count("not available") == 2 and '"fist_after_fire": "misfire"' in p2  # votes default to the segment's


@pytest.mark.parametrize(
    "text, verdict, conf, gesture",
    [
        ('{"verdict":"misfire","intent":"command","motion":"still","true_gesture":"h_left","tags":["transition"],"confidence":0.82,"reason":"r"}', Verdict.MISFIRE, 0.82, "h_left"),
        ('Sure, here it is:\n```json\n{"verdict": "intended", "intent": "command", "motion": "swipe", "true_gesture": null, "confidence": 1.4}\n```', Verdict.INTENDED, 1.0, None),
        ('{"verdict":"bogus","intent":"command","true_gesture":"wave","confidence":"high"}', Verdict.AMBIGUOUS, 0.0, None),
        ("I cannot tell.", Verdict.AMBIGUOUS, 0.0, None),
        ("", Verdict.AMBIGUOUS, 0.0, None),
    ],
)
def test_parse_answer_is_lenient(text: str, verdict: Verdict, conf: float, gesture: str | None):
    a = tagger.parse_answer(text)
    assert a.verdict is verdict and a.confidence == conf and a.true_gesture == gesture
    assert tagger.parse_answer('{"true_gesture":"none","verdict":"no_event","confidence":0.9}').true_gesture is None


def _no_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tagger, "audit_sample", lambda sid, percent=10: False)


def test_needs_human_each_condition(monkeypatch: pytest.MonkeyPatch):
    _no_audit(monkeypatch)
    # agrees with the weak label, confident, cosmos and jev agree: no human
    assert tagger.needs_human(_answer(), _seg(), _cosmos(), _jev()) == (False, [])
    # ambiguous
    assert "ambiguous" in tagger.needs_human(_answer(Verdict.AMBIGUOUS), _seg(), _cosmos(), _jev())[1]
    # under the confidence floor
    assert any("< 0.75" in w for w in tagger.needs_human(_answer(conf=0.74), _seg(), _cosmos(), _jev())[1])
    assert not tagger.needs_human(_answer(conf=0.75), _seg(), _cosmos(), _jev())[0]
    # disagrees with a strong weak label (>= 0.5); a weak one (< 0.5) or 'unsure' does not count
    assert any("weak label misfire" in w for w in tagger.needs_human(_answer(Verdict.INTENDED), _seg(), _cosmos(), _jev())[1])
    assert not tagger.needs_human(_answer(Verdict.INTENDED), _seg(weak_weight=0.49), _cosmos(), _jev())[0]
    assert not tagger.needs_human(_answer(Verdict.INTENDED), _seg(weak_label="unsure", weak_weight=1.0), _cosmos(), _jev())[0]
    # cosmos says command, jev says no sign (and the reverse); unsure cosmos or a missing pass abstains
    assert any("video intent command" in w for w in tagger.needs_human(_answer(), _seg(), _cosmos(), _jev(sign=0.1))[1])
    assert any("video intent incidental" in w for w in tagger.needs_human(_answer(), _seg(), _cosmos(Intent.INCIDENTAL), _jev(sign=0.9))[1])
    assert not tagger.needs_human(_answer(), _seg(), _cosmos(Intent.UNSURE), _jev(sign=0.1))[0]
    assert not tagger.needs_human(_answer(), _seg(), None, _jev(sign=0.1))[0]
    assert not tagger.needs_human(_answer(), _seg(), _cosmos(), None)[0]
    # every reason is listed when several apply
    human, why = tagger.needs_human(_answer(Verdict.AMBIGUOUS, conf=0.2), _seg(), _cosmos(), _jev(sign=0.1))
    assert human and len(why) == 4


def test_audit_sample_is_deterministic_and_about_ten_percent():
    ids = [f"s/fire/{i:04d}" for i in range(2000)]
    picked = [i for i in ids if tagger.audit_sample(i)]
    assert 150 <= len(picked) <= 250 and picked == [i for i in ids if tagger.audit_sample(i)]
    assert not any(tagger.audit_sample(i, percent=0) for i in ids[:50])
    # only confident, otherwise-clean tags are audited; the reason says so
    sid = picked[0]
    seg = _seg()
    seg.segment_id = sid
    human, why = tagger.needs_human(_answer(), seg, _cosmos(), _jev())
    assert human and why == ["audit sample"]
    tag = tagger.make_tag(seg, _answer(), _cosmos(), _jev(), "m")
    assert tag.needs_human and "[human: audit sample]" in tag.reason and tag.model == "m"


# -- the HTTP client ------------------------------------------------------------------


class _Resp:
    def __init__(self, body: dict[str, Any]):
        self._b = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._b


def _message(answer: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "x", "stop_reason": stop, "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": json.dumps(answer)}], "usage": {"input_tokens": 1, "output_tokens": 1}}


def _http_error(code: int, body: bytes = b'{"type":"error","error":{"type":"overloaded_error"}}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError(tagger.ANTHROPIC_URL, code, "err", None, io.BytesIO(body))  # type: ignore[arg-type]


def _tagger(urlopen: Any, **kw: Any) -> tagger.ClaudeTagger:
    kw.setdefault("sleep", lambda s: None)
    return tagger.ClaudeTagger(api_key="k", model="small", escalate_model="big", urlopen=urlopen, **kw)


GOOD = {"verdict": "misfire", "intent": "command", "motion": "still", "true_gesture": "h_left", "tags": ["held_from_previous_command"], "confidence": 0.9, "reason": "r"}


def test_request_shape_and_no_escalation_when_confident(monkeypatch: pytest.MonkeyPatch):
    _no_audit(monkeypatch)
    seen: list[Any] = []

    def urlopen(req, timeout):
        seen.append(req)
        return _Resp(_message(GOOD))

    t = _tagger(urlopen)
    tag = t.tag(_seg(), _cosmos(), _jev())
    assert t.calls == 1 and tag.model == "small" and tag.verdict is Verdict.MISFIRE and tag.true_gesture == "h_left" and not tag.needs_human
    req = seen[0]
    assert req.full_url == tagger.ANTHROPIC_URL and req.get_header("X-api-key") == "k" and req.get_header("Anthropic-version") == "2023-06-01"
    body = json.loads(req.data)
    assert body["model"] == "small" and body["system"] == tagger.SYSTEM_PROMPT and body["messages"][0]["role"] == "user" and "## Engine" in body["messages"][0]["content"]
    assert "thinking" not in body and "temperature" not in body


def test_oauth_token_uses_bearer_header():
    seen: list[Any] = []

    def urlopen(req, timeout):
        seen.append(req)
        return _Resp(_message(GOOD))

    t = tagger.ClaudeTagger(api_key="sk-ant-oat01-xyz", model="small", escalate_model="big", urlopen=urlopen)
    t.complete("small", "p")
    req = seen[0]
    assert req.get_header("Authorization") == "Bearer sk-ant-oat01-xyz" and req.get_header("Anthropic-beta") == "oauth-2025-04-20" and req.get_header("X-api-key") is None


@pytest.mark.parametrize("first", [{**GOOD, "confidence": 0.6}, {**GOOD, "verdict": "ambiguous"}, {"text": "not json"}])
def test_escalates_on_low_confidence_or_ambiguity(monkeypatch: pytest.MonkeyPatch, first: dict[str, Any]):
    _no_audit(monkeypatch)
    models: list[str] = []

    def urlopen(req, timeout):
        body = json.loads(req.data)
        models.append(body["model"])
        return _Resp(_message(first if body["model"] == "small" else GOOD))

    t = _tagger(urlopen)
    tag = t.tag(_seg(), _cosmos(), _jev())
    assert models == ["small", "big"] and tag.model == "big" and tag.confidence == 0.9 and not tag.needs_human
    # the same prompt both times
    assert t.calls == 2


def test_escalated_answer_can_still_need_a_human(monkeypatch: pytest.MonkeyPatch):
    _no_audit(monkeypatch)
    t = _tagger(lambda req, timeout: _Resp(_message({**GOOD, "confidence": 0.5})))
    tag = t.tag(_seg(), _cosmos(), _jev())
    assert t.calls == 2 and tag.needs_human and "< 0.75" in tag.reason


def test_retries_on_429_and_529_then_gives_up_on_400():
    codes = [429, 529]

    def flaky(req, timeout):
        if codes:
            raise _http_error(codes.pop(0))
        return _Resp(_message(GOOD))

    slept: list[float] = []
    t = _tagger(flaky, sleep=slept.append, backoff_s=2.0)
    assert t.complete("small", "p") == json.dumps(GOOD) and t.calls == 3 and slept == [2.0, 4.0]

    def bad(req, timeout):
        raise _http_error(400, b'{"type":"error","error":{"type":"invalid_request_error"}}')

    t = _tagger(bad)
    with pytest.raises(tagger.TaggerError) as ei:
        t.complete("small", "p")
    assert ei.value.status == 400 and t.calls == 1

    def down(req, timeout):
        raise _http_error(503)

    t = _tagger(down, max_retries=1)
    with pytest.raises(tagger.TaggerError):
        t.complete("small", "p")
    assert t.calls == 2


def test_refusal_becomes_ambiguous_and_call_cap_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _no_audit(monkeypatch)
    t = _tagger(lambda req, timeout: _Resp(_message(GOOD, stop="refusal")))
    assert t.complete("small", "p") == ""
    tag = t.tag(_seg(), None, None)
    assert tag.verdict is Verdict.AMBIGUOUS and tag.needs_human

    d = tmp_path / "intent"
    segs = [_seg(), Segment("s1/fire/0001", "s1", SegmentKind.FIRE, 5.0, 8.0), Segment("s1/hand/0000", "s1", SegmentKind.HAND, 9.0, 15.0)]
    write_jsonl(d / "segments.jsonl", segs)
    capped = _tagger(lambda req, timeout: _Resp(_message(GOOD)), max_calls=2)
    logs: list[str] = []
    n = tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", capped, log=logs.append)
    assert n == 2 and len(list(read_jsonl(d / "tags.jsonl", Tag))) == 2 and logs[-1].startswith("stopped: tagger call cap")


def test_run_tag_resumes_dry_runs_and_joins_by_id(tmp_path: Path):
    d = tmp_path / "intent"
    segs = [_seg(), Segment("s1/fire/0001", "s1", SegmentKind.FIRE, 5.0, 8.0, engine_gesture="two_up"), Segment("s1/dead/0000", "s1", SegmentKind.DEAD, 20.0, 24.0, hand_present_fraction=0.0)]
    write_jsonl(d / "segments.jsonl", segs)
    write_jsonl(d / "cosmos.jsonl", [_cosmos()])
    write_jsonl(d / "jev.jsonl", [_jev()])
    logs: list[str] = []
    fake = tagger.FakeTagger()
    assert tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", fake, dry_run=True, log=logs.append) == 3
    assert not (d / "tags.jsonl").exists() and len(fake.prompts) == 0 and "## Engine" in logs[0]
    assert tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", fake, limit=2) == 2
    assert "hand: index finger out" in fake.prompts[0] and "not available" in fake.prompts[1]  # joined by id
    assert tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", fake) == 1
    assert tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", fake) == 0  # nothing new
    tags = list(read_jsonl(d / "tags.jsonl", Tag))
    assert [t.segment_id for t in tags] == [s.segment_id for s in segs] and tags[1].true_gesture == "two_up" and tags[0].model == "fake-tagger"
    # the fake's confident 'intended' disagrees with the strong 'misfire' weak label of the first segment
    assert tags[0].needs_human and "weak label misfire" in tags[0].reason
