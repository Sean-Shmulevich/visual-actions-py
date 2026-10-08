from visual_actions.tools.calibrate import pinch_thresholds, reach_box
from visual_actions.tools.synth import hand_frame


def test_reach_box_covers_where_the_hand_went():
    hfs = [hand_frame("open_palm", i, center=(0.3 + 0.4 * (i % 11) / 10, 0.4 + 0.2 * (i % 7) / 6), mirror_to_raw=False) for i in range(200)]
    x0, x1, y0, y1 = reach_box(hfs, pad=0.0)
    assert 0.25 <= x0 <= 0.35 and 0.65 <= x1 <= 0.75
    assert 0.35 <= y0 <= 0.45 and 0.55 <= y1 <= 0.65


def test_pinch_thresholds_separate_the_two_distributions():
    pinch = [hand_frame("pinch", i, mirror_to_raw=False) for i in range(50)]
    other = [hand_frame(n, i, mirror_to_raw=False) for i in range(20) for n in ("open_palm", "fist", "h_left", "point_up")]
    on, off, stats = pinch_thresholds(pinch, other)
    assert stats["pinch_p90"] < on <= 0.45
    assert on < off
    assert off - on >= 0.12
