from visual_actions.core.types import Hand, HandFrame, Landmark
from visual_actions.tools.synth import hand_frame
from visual_actions.ui.capture import CaptureThread


def shifted(dx, dy):
    hf = hand_frame("open_palm", 0, center=(0.5, 0.5), mirror_to_raw=False)
    return HandFrame(0, Hand.RIGHT, tuple(Landmark(lm.x + dx, lm.y + dy, lm.z) for lm in hf.landmarks), 0.9)


def test_centred_hand_is_in_frame():
    assert CaptureThread.out_of_frame(shifted(0, 0)) is None


def test_wrist_outside_is_lost():
    assert "wrist outside" in (CaptureThread.out_of_frame(shifted(0.6, 0)) or "")


def test_fingers_above_the_top_edge_is_lost_even_with_wrist_inside():
    reason = CaptureThread.out_of_frame(shifted(0, -0.42))
    assert reason is not None and ("landmarks outside" in reason or "pinch point" in reason)


def test_reaching_down_with_wrist_below_the_frame_is_still_in_view():
    # wrist well below the bottom edge, fingers and pinch point in view: the hand is usable
    assert CaptureThread.out_of_frame(shifted(0, 0.45)) is None


def test_pinch_point_below_the_frame_is_lost():
    assert CaptureThread.out_of_frame(shifted(0, 0.75)) is not None
