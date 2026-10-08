import math

from visual_actions.core.pointer import OneEuroFilter, PointerMap, ReachBox, SmoothedPointer


def test_reach_box_maps_to_full_screen_and_clamps():
    m = PointerMap(1440, 900, ReachBox(0.2, 0.8, 0.2, 0.8))
    close = lambda a, b: all(math.isclose(p, q, abs_tol=1e-6) for p, q in zip(a, b))
    assert close(m.to_screen(0.2, 0.2), (0.0, 0.0))
    assert close(m.to_screen(0.8, 0.8), (1440.0, 900.0))
    assert close(m.to_screen(0.5, 0.5), (720.0, 450.0))
    assert close(m.to_screen(0.0, 1.0), (0.0, 900.0))  # outside the box clamps to the edge


def test_user_right_is_screen_right():
    m = PointerMap(1000, 1000)
    assert m.to_screen(0.7, 0.5)[0] > m.to_screen(0.3, 0.5)[0]


def test_depth_normalization_shrinks_box_for_a_far_hand():
    m = PointerMap(1000, 1000, ReachBox(0.2, 0.8, 0.2, 0.8), depth_gain=1.0, ref_hand_scale=0.12)
    near = m.effective_box(0.12)
    far = m.effective_box(0.06)
    assert abs(near.x0 - 0.2) < 1e-9
    assert far.x1 - far.x0 < near.x1 - near.x0
    assert abs((far.x0 + far.x1) / 2 - 0.5) < 1e-9  # same centre
    # the same physical reach (half the frame displacement at half the size) lands on the same pixel
    assert abs(m.to_screen(0.5 + 0.15, 0.5, 0.12)[0] - m.to_screen(0.5 + 0.075, 0.5, 0.06)[0]) < 1e-6


def test_depth_gain_zero_ignores_hand_size():
    m = PointerMap(1000, 1000, depth_gain=0.0)
    assert m.to_screen(0.6, 0.6, 0.05) == m.to_screen(0.6, 0.6, 0.2)


def test_one_euro_filter_tracks_a_ramp_and_damps_jitter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = [f(float(i), i / 30) for i in range(60)]
    assert out[-1] > 50  # follows a ramp with bounded lag
    g = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    noisy = [100 + (5 if i % 2 else -5) for i in range(60)]
    smoothed = [g(v, i / 30) for i, v in enumerate(noisy)]
    assert max(abs(s - 100) for s in smoothed[10:]) < 3


def test_smoothed_pointer_reset_and_first_sample():
    p = SmoothedPointer(PointerMap(1000, 1000), min_cutoff=1.0, beta=0.0)
    x, y = p.update(0, 0.5, 0.5)
    assert math.isclose(x, 500) and math.isclose(y, 500)
    p.update(33_000_000, 0.9, 0.9)
    p.reset()
    x2, _ = p.update(66_000_000, 0.5, 0.5)
    assert math.isclose(x2, 500)
