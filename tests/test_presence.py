from visual_actions.core.presence import PresenceFilter
from visual_actions.core.types import Hand, HandFrame, Landmark
from visual_actions.tools.synth import hand_frame


def hand(dx=0.0, dy=0.0):
    hf = hand_frame("open_palm", 0, center=(0.5, 0.5), mirror_to_raw=False)
    return HandFrame(0, Hand.RIGHT, tuple(Landmark(lm.x + dx, lm.y + dy, lm.z) for lm in hf.landmarks), 0.9)


def test_one_or_two_missing_frames_do_not_lose_the_hand():
    f = PresenceFilter(lost_frames=3)
    assert f.update(hand()).seen is not None
    assert f.update(None).lost is None
    assert f.update(None).lost is None
    assert f.update(hand()).seen is not None and f.present


def test_three_missing_frames_lose_the_hand_with_tracker_reason():
    f = PresenceFilter(lost_frames=3)
    f.update(hand())
    f.update(None)
    f.update(None)
    pr = f.update(None, frame_index=42)
    assert pr.lost is not None and pr.lost[0] == "tracker" and "42" in pr.lost[1]
    assert not f.present
    assert f.update(None).lost is None  # already lost: no repeat


def test_edge_frames_count_as_missing_and_carry_the_edge_reason():
    f = PresenceFilter(lost_frames=2)
    f.update(hand())
    assert f.update(hand(dx=0.6)).lost is None  # wrist outside: first missing frame
    pr = f.update(hand(dx=0.6))
    assert pr.lost is not None and pr.lost[0] == "edge" and "wrist outside" in pr.lost[1]


def test_gate_closure_is_immediate():
    f = PresenceFilter()
    f.update(hand())
    pr = f.gate_closed()
    assert pr.lost is not None and pr.lost[0] == "gate"
    assert f.gate_closed().lost is None


def test_never_seen_hand_never_reports_lost():
    f = PresenceFilter(lost_frames=1)
    assert f.update(None).lost is None and f.gate_closed().lost is None
