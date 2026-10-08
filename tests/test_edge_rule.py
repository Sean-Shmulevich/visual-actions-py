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


def test_many_landmarks_outside_is_lost_even_with_wrist_inside():
    reason = CaptureThread.out_of_frame(shifted(0, -0.42))  # fingers above the top edge, wrist still inside
    assert reason is not None and "landmarks outside" in reason
