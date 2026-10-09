"""events.log -> typed events -> segments with weak labels."""

from pathlib import Path

from visual_actions.intent.events import parse_events, session_t0_ns
from visual_actions.intent.labelfns import ARM_FNS, FIRE_FNS, vote
from visual_actions.intent.schema import Intent, Motion, Segment, SegmentKind, Tag, Verdict, by_id, read_jsonl, write_jsonl
from visual_actions.intent.segments import segment_session

LOG = """\
    0.000  23:12:52.309591  session started 2026-10-08 23:12:52 t0_ns=5000000000
    1.000  23:12:53.000000  hand    seen conf=0.99 wrist=(0.50,0.50) (mode idle)
    1.250  23:12:53.250000  token   open_palm conf=0.97 still=True
    1.250  23:12:53.250001  mode    idle -> holding [window]
    2.800  23:12:54.800000  mode    holding -> armed [window]
    3.300  23:12:55.300000  token   h_left conf=0.95 still=True
    3.300  23:12:55.300001  action  Cmd+Tab ok
    3.300  23:12:55.300002  mode    armed -> armed [window]
    3.800  23:12:55.800000  token   fist conf=0.90 still=True
    3.800  23:12:55.800001  mode    armed -> idle
   12.000  23:13:04.000000  hand    lost [tracker] tracker reported no hand (frame 300) (mode idle)
   25.000  23:13:17.000000  hand    seen after 13.00s away conf=0.95 wrist=(0.40,0.60) (mode idle)
   25.250  23:13:17.250000  token   open_palm conf=0.96 still=True
   25.250  23:13:17.250001  mode    idle -> holding [window]
   26.800  23:13:18.800000  mode    holding -> armed [window]
   27.300  23:13:19.300000  token   two_up conf=0.93 still=True
   27.300  23:13:19.300001  action  Next tab ok
   27.300  23:13:19.300002  mode    armed -> repeat [window]
   27.900  23:13:19.900000  token   two_up conf=0.92 still=False
   27.900  23:13:19.900001  mode    repeat -> repeat [window]
   27.900  23:13:19.900002  action  Next tab ok
   29.500  23:13:21.500000  mode    repeat -> armed [window]
   34.500  23:13:26.500000  mode    armed -> idle
   34.600  23:13:26.600000  hand    lost [edge] wrist outside (mode idle)
"""


def events(tmp_path: Path):
    p = tmp_path / "events.log"
    p.write_text(LOG)
    return parse_events(p)


def test_events_parse_fields(tmp_path: Path):
    ev = events(tmp_path)
    assert ev[0].kind == "session" and session_t0_ns(ev, tmp_path / "nope.jsonl") == 5_000_000_000
    tok = ev[2]
    assert tok.token == "open_palm" and tok.confidence == 0.97 and tok.still is True
    mode = ev[3]
    assert (mode.mode_from, mode.mode_to, mode.namespace) == ("idle", "holding", "window")
    act = ev[6]
    assert act.action_name == "Cmd+Tab" and act.action_ok is True
    assert ev[1].hand_seen is True and ev[10].hand_seen is False


def test_t0_falls_back_to_first_seen_frame(tmp_path: Path):
    p = tmp_path / "events.log"
    p.write_text(LOG.replace(" t0_ns=5000000000", ""))
    (tmp_path / "landmarks.jsonl").write_text('{"t_ns": 6000000000, "hand": "right", "confidence": 1.0, "landmarks": []}\n')
    assert session_t0_ns(parse_events(p), tmp_path / "landmarks.jsonl") == 5_000_000_000  # 6 s frame seen at elapsed 1.0


def test_labelling_functions_vote(tmp_path: Path):
    ev = events(tmp_path)
    i_fire = next(i for i, e in enumerate(ev) if e.kind == "action")
    votes, label, weight = vote(ev, i_fire, FIRE_FNS)
    assert votes["fist_after_fire"] == "misfire" and label == "misfire" and weight > 0
    i_arm = next(i for i, e in enumerate(ev) if e.kind == "mode" and e.mode_to == "armed")
    votes, label, _ = vote(ev, i_arm, ARM_FNS)
    assert votes == {"arm_then_command": "intended"} and label == "intended"


def test_segments_cover_each_engine_event_and_the_gaps(tmp_path: Path):
    ev = events(tmp_path)
    segs = segment_session("s1", ev)
    kinds = [s.kind for s in segs]
    assert kinds.count(SegmentKind.ARM) == 2 and kinds.count(SegmentKind.FIRE) == 3 and kinds.count(SegmentKind.REPEAT) == 1
    assert kinds.count(SegmentKind.DEAD) == 1  # the 13 s gap, sampled once
    fire = next(s for s in segs if s.kind == SegmentKind.FIRE)
    assert fire.engine_gesture == "h_left" and fire.engine_action == "Cmd+Tab" and fire.engine_outcome == "cancelled_by_fist"
    assert fire.weak_label == "misfire" and fire.t0 == 1.8 and fire.t1 == 4.8
    arm = next(s for s in segs if s.kind == SegmentKind.ARM)
    assert arm.engine_gesture == "open_palm" and arm.engine_outcome == "command" and arm.engine_namespace == "window"
    assert 0.9 < arm.hand_present_fraction <= 1.0
    dead = next(s for s in segs if s.kind == SegmentKind.DEAD)
    assert dead.hand_present_fraction == 0.0 and 12.0 <= dead.t0 < dead.t1 <= 25.0
    rep = next(s for s in segs if s.kind == SegmentKind.REPEAT)
    assert rep.engine_outcome == "fires=1" and rep.engine_gesture == "two_up"
    ids = [s.segment_id for s in segs]
    assert len(ids) == len(set(ids)) and all(i.startswith("s1/") for i in ids)


def test_jsonl_round_trip_keeps_enums(tmp_path: Path):
    seg = Segment("s/fire/0000", "s", SegmentKind.FIRE, 1.0, 2.0, engine_gesture="h_left", labelfn_votes={"x": "misfire"})
    tag = Tag("s/fire/0000", Verdict.MISFIRE, Intent.INCIDENTAL, Motion.STILL, None, tags=["face_touch"], confidence=0.8)
    write_jsonl(tmp_path / "segments.jsonl", [seg])
    write_jsonl(tmp_path / "tags.jsonl", [tag, Tag("s/fire/0000", Verdict.INTENDED, Intent.COMMAND, Motion.STILL, "h_left")])
    back = list(read_jsonl(tmp_path / "segments.jsonl", Segment))
    assert back == [seg] and back[0].kind is SegmentKind.FIRE
    tags = by_id(read_jsonl(tmp_path / "tags.jsonl", Tag))
    assert tags["s/fire/0000"].verdict is Verdict.INTENDED and tags["s/fire/0000"].intent is Intent.COMMAND  # last wins
