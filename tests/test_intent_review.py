"""The human review queue: join, ordering, cap, and the one-key server."""

import http.client
import json
from pathlib import Path

import pytest

from visual_actions.intent.review import (
    AUDIT,
    DISAGREEMENT,
    HIGH_MISFIRE,
    REST,
    ReviewServer,
    build_queue,
    clip_name,
    human_path,
)
from visual_actions.intent.schema import (
    CosmosVerdict,
    HumanLabel,
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


def seg(i: int, kind: SegmentKind = SegmentKind.FIRE, weak: str | None = None, weight: float = 0.0, gesture: str | None = "h_left", t0: float | None = None) -> Segment:
    t = float(i * 10) if t0 is None else t0
    return Segment(f"s1/{kind.value}/{i:04d}", "s1", kind, t, t + 3.0, engine_gesture=gesture, engine_action="Cmd+Tab" if kind is SegmentKind.FIRE else None, mean_confidence=0.9, hand_present_fraction=0.0 if kind is SegmentKind.DEAD else 1.0, labelfn_votes={"fist_after_fire": weak} if weak == "misfire" else {}, weak_label=weak, weak_weight=weight)


def tag(s: Segment, verdict: Verdict, conf: float, needs_human: bool, gesture: str | None = "h_left") -> Tag:
    return Tag(s.segment_id, verdict, Intent.COMMAND if verdict is Verdict.INTENDED else Intent.INCIDENTAL, Motion.STILL, gesture, confidence=conf, needs_human=needs_human, reason="test")


def write_session(tmp_path: Path, segments: list[Segment], tags: list[Tag] = (), cosmos: list[CosmosVerdict] = (), jev: list[JevVerdict] = (), human: list[HumanLabel] = ()) -> Path:
    d = tmp_path / "s1"
    write_jsonl(d / "intent" / "segments.jsonl", segments)
    if tags:
        write_jsonl(d / "intent" / "tags.jsonl", tags)
    if cosmos:
        write_jsonl(d / "intent" / "cosmos.jsonl", cosmos)
    if jev:
        write_jsonl(d / "intent" / "jev.jsonl", jev)
    if human:
        write_jsonl(d / "intent" / "human.jsonl", human)
    return d


def test_queue_joins_passes_and_applies_the_keep_rules(tmp_path: Path):
    s_needs = seg(0, weak="intended", weight=0.7)  # tag asks for a human
    s_confident = seg(1, weak="intended", weight=0.7)  # tag confident, agrees: not queued (audit aside)
    s_untagged_misfire = seg(2, weak="misfire", weight=0.3)  # no tag, weak says misfire
    s_untagged_fire = seg(3, weak="intended", weight=0.7)  # fire with no tag at all
    s_hand_untagged = seg(4, kind=SegmentKind.HAND, gesture=None)  # hand, no tag, no weak label: not queued
    s_arm_untagged = seg(5, kind=SegmentKind.ARM, gesture="open_palm")
    cosmos = [CosmosVerdict(s_needs.segment_id, True, True, "yes", True, False, Intent.COMMAND, Motion.STILL, "flat hand", "looks deliberate", 0.8)]
    jev = [JevVerdict(s_needs.segment_id, {"q1": 0.9}, choice="h_left", choice_probs={"h_left": 0.9}, confidence=0.9)]
    d = write_session(tmp_path, [s_needs, s_confident, s_untagged_misfire, s_untagged_fire, s_hand_untagged, s_arm_untagged], [tag(s_needs, Verdict.INTENDED, 0.5, True), tag(s_confident, Verdict.INTENDED, 0.95, False)], cosmos, jev)
    q = build_queue(d, audit_fraction=0.0)
    ids = {it.segment_id for it in q}
    assert ids == {s_needs.segment_id, s_untagged_misfire.segment_id, s_untagged_fire.segment_id, s_arm_untagged.segment_id}
    first = next(it for it in q if it.segment_id == s_needs.segment_id)
    assert first.cosmos is not None and first.cosmos.hand_description == "flat hand"
    assert first.jev is not None and first.jev.choice == "h_left"
    assert first.tag is not None and first.reason.startswith("judge asked")
    j = first.to_json()
    assert j["segment"]["engine_gesture"] == "h_left" and j["cosmos"]["intent"] == "command" and j["bucket_name"]


def test_queue_orders_disagreement_then_misfires_then_audit_then_rest(tmp_path: Path):
    s_rest = seg(0, weak="unsure")
    s_dis = seg(1, weak="intended", weight=0.7)
    s_mis = seg(2, weak="misfire", weight=0.6)
    s_audit = seg(3, weak="intended", weight=0.7)
    s_mis_tag = seg(4, weak="misfire", weight=0.6)
    tags = [tag(s_dis, Verdict.MISFIRE, 0.9, False), tag(s_audit, Verdict.INTENDED, 0.95, False), tag(s_mis_tag, Verdict.MISFIRE, 0.85, True)]
    d = write_session(tmp_path, [s_rest, s_dis, s_mis, s_audit, s_mis_tag], tags)
    q = build_queue(d, audit_fraction=1.0)
    assert [it.bucket for it in q] == [DISAGREEMENT, HIGH_MISFIRE, HIGH_MISFIRE, AUDIT, REST]
    assert q[0].segment_id == s_dis.segment_id and "disagree" in q[0].bucket_name
    assert q[1].segment_id == s_mis_tag.segment_id  # tag confidence 0.85 beats weak weight 0.6
    assert q[2].segment_id == s_mis.segment_id
    assert q[3].segment_id == s_audit.segment_id and q[4].segment_id == s_rest.segment_id
    # the audit sample is deterministic and off when the fraction is 0
    assert s_audit.segment_id not in {it.segment_id for it in build_queue(d, audit_fraction=0.0)}


def test_queue_caps_lowest_value_first_and_skips_labelled_ids(tmp_path: Path):
    segs = [seg(i, weak="misfire", weight=0.2 + 0.1 * i) for i in range(6)]
    human = [HumanLabel(segs[5].segment_id, Verdict.MISFIRE, Intent.INCIDENTAL, at="2026-10-09T10:00:00")]
    d = write_session(tmp_path, segs, human=human)
    q = build_queue(d, cap=3)
    assert [it.segment_id for it in q] == [segs[4].segment_id, segs[3].segment_id, segs[2].segment_id]
    assert [it.bucket for it in q] == [HIGH_MISFIRE, HIGH_MISFIRE, REST]  # 0.4 is under the confident-misfire bar
    assert all(it.segment_id != segs[5].segment_id for it in build_queue(d))


def test_clip_name_flattens_the_id():
    assert clip_name("20261008-231252/fire/0003") == "20261008-231252_fire_0003"


# -- the server ---------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path):
    segs = [seg(0, weak="misfire", weight=0.6), seg(1, weak="misfire", weight=0.5), seg(2, kind=SegmentKind.DEAD, gesture=None, weak="unsure")]
    d = write_session(tmp_path, segs)
    srv = ReviewServer(d, port=0)
    try:
        yield srv
    finally:
        srv.close()


def call(srv: ReviewServer, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=3)
    c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else {"raw": data}


def test_page_and_item_endpoint(server: ReviewServer):
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
    c.request("GET", "/")
    page = c.getresponse().read().decode()
    assert "<title>Intent review</title>" in page and "'/label'" in page and "'/undo'" in page
    for key in ("y", "n", "m", "x", "a", "g", "u", "q"):
        assert f"<b>{key}</b>" in page or f"'{key}'" in page
    status, s = call(server, "GET", "/item")
    assert status == 200 and s["total"] == 3 and s["index"] == 0 and s["done"] == 0
    assert s["item"]["segment"]["segment_id"] == "s1/fire/0000" and s["item"]["reason"].startswith("labelfns: misfire")
    assert s["legend"]["p"] == "open_palm" and s["legend"]["0"] == "none"
    assert [o["segment_id"] for o in s["order"]][:2] == ["s1/fire/0000", "s1/fire/0001"]
    assert s["media"]  # keys present whether or not clips.py is importable
    status, _ = call(server, "GET", "/nope")
    assert status == 404


def test_label_appends_advances_and_undo_removes(server: ReviewServer):
    hp = human_path(server.session_dir)
    status, s = call(server, "POST", "/label", {"segment_id": "s1/fire/0000", "verdict": "intended"})
    assert status == 200 and s["index"] == 1 and s["done"] == 1 and s["item"]["segment"]["segment_id"] == "s1/fire/0001"
    labels = list(read_jsonl(hp, HumanLabel))
    assert len(labels) == 1 and labels[0].verdict is Verdict.INTENDED and labels[0].intent is Intent.COMMAND
    assert labels[0].true_gesture == "h_left" and labels[0].at and "T" in labels[0].at  # intended defaults to the engine's gesture, ISO stamp

    # g + letter amends the last answer's gesture: one more write, still one line
    status, s = call(server, "POST", "/label", {"segment_id": "s1/fire/0000", "verdict": "intended", "true_gesture": "fist", "amend": True})
    labels = list(read_jsonl(hp, HumanLabel))
    assert status == 200 and len(labels) == 1 and labels[0].true_gesture == "fist" and s["index"] == 1 and s["done"] == 1
    assert s["last"]["true_gesture"] == "fist"

    status, s = call(server, "POST", "/label", {"segment_id": "s1/fire/0001", "verdict": "misfire"})
    assert s["index"] == 2 and s["done"] == 2
    labels = list(read_jsonl(hp, HumanLabel))
    assert [l.verdict for l in labels] == [Verdict.INTENDED, Verdict.MISFIRE] and labels[1].intent is Intent.INCIDENTAL and labels[1].true_gesture is None

    status, s = call(server, "POST", "/undo")
    assert status == 200 and s["index"] == 1 and s["done"] == 1 and s["item"]["segment"]["segment_id"] == "s1/fire/0001"
    labels = list(read_jsonl(hp, HumanLabel))
    assert len(labels) == 1 and labels[0].segment_id == "s1/fire/0000"

    # no-event on a dead segment is intent dead; skip and the end of the queue
    status, s = call(server, "POST", "/skip")
    assert s["index"] == 2
    status, s = call(server, "POST", "/label", {"segment_id": "s1/dead/0002", "verdict": "no_event"})
    assert s["item"] is None and s["remaining"] == 0
    assert list(read_jsonl(hp, HumanLabel))[-1].intent is Intent.DEAD

    status, s = call(server, "POST", "/label", {"segment_id": "nope", "verdict": "intended"})
    assert status == 400 and "error" in s
    status, s = call(server, "POST", "/label", {"segment_id": "s1/fire/0001", "verdict": "bogus"})
    assert status == 400


def test_undo_with_nothing_labelled_is_a_noop(server: ReviewServer):
    status, s = call(server, "POST", "/undo")
    assert status == 200 and s["index"] == 0 and s["done"] == 0 and not human_path(server.session_dir).exists()


def test_media_falls_back_to_text_when_there_is_no_video(server: ReviewServer):
    # a session folder with no video: the clip endpoint answers 404 (or 200 if clips.py can cut), never 500
    status, _ = call(server, "GET", "/clip/s1_fire_0000.mp4")
    assert status in (200, 404)
    status, _ = call(server, "GET", "/strip/s1_fire_0000.png")
    assert status in (200, 404)
    status, _ = call(server, "GET", "/clip/unknown.mp4")
    assert status == 404
