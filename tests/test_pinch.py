from visual_actions.core.pinch import (
    PinchDetector,
    PinchPhase,
    hand_scale,
    pinch_distance,
    pinch_point,
)
from visual_actions.tools.synth import hand_frame


def test_pinch_distance_separates_pinch_from_open_palm():
    assert pinch_distance(hand_frame("pinch", 0, mirror_to_raw=False)) < 0.3
    assert pinch_distance(hand_frame("open_palm", 0, mirror_to_raw=False)) > 1.0
    assert pinch_distance(hand_frame("fist", 0, mirror_to_raw=False)) > 0.35  # a fist is not a pinch


def test_pinch_distance_is_scale_and_position_invariant():
    a = pinch_distance(hand_frame("pinch", 0, center=(0.3, 0.3), scale=0.08, mirror_to_raw=False))
    b = pinch_distance(hand_frame("pinch", 0, center=(0.7, 0.6), scale=0.2, mirror_to_raw=False))
    assert abs(a - b) < 1e-9


def test_pinch_point_and_scale():
    hf = hand_frame("pinch", 0, center=(0.5, 0.5), scale=0.1, mirror_to_raw=False)
    x, y = pinch_point(hf)
    assert 0.3 < x < 0.7 and 0.2 < y < 0.6
    assert abs(hand_scale(hf) - 0.1) < 1e-9


def test_detector_hysteresis_and_debounce():
    det = PinchDetector(on_threshold=0.3, off_threshold=0.5, debounce_frames=2, release_frames=2)
    frames = [hand_frame("open_palm", i * 33_000_000, mirror_to_raw=False) for i in range(3)]
    frames += [hand_frame("pinch", (3 + i) * 33_000_000, mirror_to_raw=False) for i in range(5)]
    frames += [hand_frame("open_palm", (8 + i) * 33_000_000, mirror_to_raw=False) for i in range(3)]
    phases = [ev.phase for ev in (det.update(f) for f in frames) if ev is not None]
    # 2-frame debounce: START on the 2nd pinch frame, MOVE on the 3 after, END on the 2nd open frame
    assert phases[0] is PinchPhase.START
    assert phases[-1] is PinchPhase.END
    assert phases.count(PinchPhase.START) == 1 and phases.count(PinchPhase.END) == 1
    assert phases.count(PinchPhase.MOVE) == 3


def test_single_noisy_frame_does_not_toggle():
    det = PinchDetector(debounce_frames=2)
    for i in range(4):
        det.update(hand_frame("pinch", i * 33_000_000, mirror_to_raw=False))
    assert det.pinched
    ev = det.update(hand_frame("open_palm", 5 * 33_000_000, mirror_to_raw=False))
    assert det.pinched and ev is None  # pending release: hold position, no move
    ev = det.update(hand_frame("pinch", 6 * 33_000_000, mirror_to_raw=False))
    assert det.pinched and ev is not None and ev.phase is PinchPhase.MOVE


def test_end_event_when_hand_is_lost_mid_pinch():
    det = PinchDetector(debounce_frames=1)
    det.update(hand_frame("pinch", 0, mirror_to_raw=False))
    assert det.pinched
    ev = det.end_event(10)
    assert ev is not None and ev.phase is PinchPhase.END and not det.pinched
    assert det.end_event(20) is None
