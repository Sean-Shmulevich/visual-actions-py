import numpy as np

from visual_actions.core.normalize import N_FEATURES, canonical, features, to_user_frame
from visual_actions.core.types import INDEX_TIP, MIDDLE_MCP, WRIST, Hand, HandFrame, Landmark
from visual_actions.tools.synth import hand_frame


def _shift_scale(hf: HandFrame, dx: float, dy: float, s: float) -> HandFrame:
    return HandFrame(
        hf.t_ns,
        hf.hand,
        tuple(Landmark(0.5 + (lm.x - 0.5) * s + dx, 0.5 + (lm.y - 0.5) * s + dy, lm.z) for lm in hf.landmarks),
        hf.confidence,
    )


def test_feature_length():
    hf = hand_frame("open_palm", 0, mirror_to_raw=False)
    assert features(hf).shape == (N_FEATURES,)


def test_translation_and_scale_invariance():
    a = hand_frame("h_left", 0, mirror_to_raw=False)
    b = _shift_scale(a, 0.1, -0.05, 1.7)
    assert np.allclose(features(a), features(b), atol=1e-6)


def test_mirror_flips_x_only():
    raw = hand_frame("h_left", 0, mirror_to_raw=True)
    user = to_user_frame(raw, mirror=True)
    for r, u in zip(raw.landmarks, user.landmarks):
        assert abs((1.0 - r.x) - u.x) < 1e-9
        assert r.y == u.y


def test_handedness_label_does_not_change_geometry():
    # MediaPipe's handedness label is unreliable; canonical() must ignore it.
    r = canonical(hand_frame("h_left", 0, hand=Hand.RIGHT, mirror_to_raw=False))
    pts = hand_frame("h_left", 0, hand=Hand.RIGHT, mirror_to_raw=False).landmarks
    relabelled = canonical(HandFrame(0, Hand.LEFT, pts, 1.0))
    assert np.allclose(r.pts, relabelled.pts, atol=1e-9)


def test_direction_ignores_handedness_label():
    pts = hand_frame("h_right", 0, hand=Hand.RIGHT, mirror_to_raw=False).landmarks
    f_right = features(HandFrame(0, Hand.RIGHT, pts, 1.0))
    f_left_label = features(HandFrame(0, Hand.LEFT, pts, 1.0))
    assert f_right[-2] > 0 and f_left_label[-2] > 0


def test_direction_feature_distinguishes_h_left_from_h_right():
    fl = features(hand_frame("h_left", 0, mirror_to_raw=False))
    fr = features(hand_frame("h_right", 0, mirror_to_raw=False))
    assert fl[-2] < 0 < fr[-2]  # cos of index direction


def test_canonical_origin_and_scale():
    c = canonical(hand_frame("open_palm", 0, mirror_to_raw=False))
    assert np.allclose(c.pts[WRIST], 0)
    assert abs(np.linalg.norm(c.pts[MIDDLE_MCP][:2]) - 1.0) < 1e-9
    assert c.pts[INDEX_TIP][1] < 0  # up is -y
