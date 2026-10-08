import random

import numpy as np

from visual_actions.core.normalize import canonical, to_user_frame
from visual_actions.core.recognizer import H_LEFT, H_RIGHT, RuleRecognizer
from visual_actions.core.recorder import read_session
from visual_actions.core.types import INDEX_TIP, MIDDLE_MCP, N_LANDMARKS, THUMB_TIP, Hand, HandFrame
from visual_actions.tools import synth3d as S

BONES = [
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _lengths(pts):
    return np.array([np.linalg.norm(pts[b] - pts[a]) for a, b in BONES])


def test_landmark_order_and_count():
    pts = S.forward_kinematics(S.HandPose())
    assert pts.shape == (N_LANDMARKS, 3)
    assert np.allclose(pts[0], 0)  # wrist at the origin
    # MCPs run thumb side (-x) to little-finger side (+x) in MediaPipe order 5, 9, 13, 17
    assert pts[5, 0] < pts[9, 0] < pts[13, 0] < pts[17, 0]
    # fingertips are the last point of each finger, farther from the wrist than the MCP when extended
    for base in (5, 9, 13, 17):
        assert np.linalg.norm(pts[base + 3]) > np.linalg.norm(pts[base])


def test_bone_lengths_preserved_under_rotation_and_flexion():
    rng = random.Random(0)
    ref = _lengths(S.forward_kinematics(S.HandPose()))
    for cls in ("open_palm", "fist", "pinch", "none"):
        pts = S.forward_kinematics(S.POSE_FUNCS[cls](rng))
        assert np.allclose(_lengths(pts), ref, atol=1e-9), cls
        view = S.View(yaw=37, pitch=-20, roll=65, depth=1e9, scale=1.0, center=(0, 0), jitter=0.0)
        rot = S.render(pts, view, None)  # orthographic at infinite depth; x mirrored, y flipped
        # mirrored/flipped projection keeps in-plane bone lengths only for pure rolls, so compare the
        # 3D rotation directly instead
        R = (
            S._rot(np.array([0, 0, 1.0]), 65)
            @ S._rot(np.array([0, 1.0, 0]), 37)
            @ S._rot(np.array([1.0, 0, 0]), -20)
        )
        assert np.allclose(_lengths(pts @ R.T), ref, atol=1e-9)
        assert rot.shape == (21, 3)


def test_left_hand_is_a_mirror():
    pose = S.pose_open_palm(random.Random(1))
    right = S.forward_kinematics(pose)
    pose.hand = Hand.LEFT
    left = S.forward_kinematics(pose)
    assert np.allclose(left[:, 0], -right[:, 0]) and np.allclose(left[:, 1:], right[:, 1:])


def test_pinch_touches_and_open_palm_does_not():
    rng = random.Random(2)
    for _ in range(20):
        p = S.forward_kinematics(S.pose_pinch(rng))
        scale = np.linalg.norm(p[MIDDLE_MCP])
        assert np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP]) / scale < 0.35
        o = S.forward_kinematics(S.pose_open_palm(rng))
        # real open palms measure ~0.8 here (MediaPipe thumb tip to index tip over wrist->middle MCP)
        assert np.linalg.norm(o[THUMB_TIP] - o[INDEX_TIP]) / np.linalg.norm(o[MIDDLE_MCP]) > 0.6


def test_rendered_frames_are_raw_camera_frame_and_in_range():
    rng = random.Random(3)
    for cls in S.STATIC_CLASSES:
        hf = S.sample(cls, rng)
        assert len(hf.landmarks) == N_LANDMARKS
        xs = [lm.x for lm in hf.landmarks]
        ys = [lm.y for lm in hf.landmarks]
        assert -0.3 < min(xs) and max(xs) < 1.3 and -0.3 < min(ys) and max(ys) < 1.3
    # raw frame: an H pointing to the user's LEFT has its index tip at LARGER raw x than the wrist
    for _ in range(10):
        hf = S.sample("h_left", rng)
        assert hf.landmarks[INDEX_TIP].x > hf.landmarks[0].x
        hf = S.sample("h_right", rng)
        assert hf.landmarks[INDEX_TIP].x < hf.landmarks[0].x


def test_rules_agree_on_synthetic_h_signs_after_mirroring():
    rng = random.Random(4)
    r = RuleRecognizer()
    hits = {H_LEFT: 0, H_RIGHT: 0}
    for cls in (H_LEFT, H_RIGHT):
        for _ in range(40):
            got, _ = r.classify(to_user_frame(S.sample(cls, rng), mirror=True))
            hits[cls] += got == cls
    # the hand-written rules are strict (thumb, curl thresholds); a clear minority is enough here
    assert hits[H_LEFT] >= 10 and hits[H_RIGHT] >= 10, hits


def test_sequence_monotonic_with_pinch_segment():
    frames = S.generate_sequence("pinch_drag", 3.0, rng=random.Random(5))
    ts = [f.t_ns for f in frames]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert len(frames) == 90
    d = []
    for hf in frames:
        c = canonical(to_user_frame(hf, True))
        d.append(float(np.linalg.norm(c.pts[THUMB_TIP][:2] - c.pts[INDEX_TIP][:2])))
    pinched = [x < 0.35 for x in d]
    assert not pinched[0] and any(pinched) and sum(pinched) > 20
    # the hand moves while pinched
    xs = [hf.landmarks[0].x for hf, p in zip(frames, pinched) if p]
    assert max(xs) - min(xs) > 0.05


def test_other_sequences():
    palm = S.generate_sequence("palm_hold", 1.0, rng=random.Random(6))
    none = S.generate_sequence("none_motion", 1.0, rng=random.Random(7))
    assert len(palm) == 30 and len(none) == 30
    assert all(isinstance(f, HandFrame) for f in palm + none)


def test_deterministic_with_seed(tmp_path):
    a = S.sample("fist", random.Random(9)).landmarks
    b = S.sample("fist", random.Random(9)).landmarks
    assert a == b
    S.write_static(tmp_path, per_class=5, seed=11, classes=("open_palm", "pinch"))
    S.write_static(tmp_path / "again", per_class=5, seed=11, classes=("open_palm", "pinch"))
    first = list(read_session(tmp_path / "open_palm" / "synth3d-11.jsonl"))
    second = list(read_session(tmp_path / "again" / "open_palm" / "synth3d-11.jsonl"))
    assert first == second and len(first) == 5
    assert (tmp_path / "no_pinch" / "synth3d-11.jsonl").exists()
