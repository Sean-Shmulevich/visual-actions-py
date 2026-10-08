import pytest

from visual_actions.core.recognizer import (
    FIST,
    H_LEFT,
    H_RIGHT,
    NONE,
    OPEN_PALM,
    RuleRecognizer,
    Smoother,
)
from visual_actions.core.types import Hand
from visual_actions.tools.synth import hand_frame


@pytest.mark.parametrize("name", [OPEN_PALM, FIST, H_LEFT, H_RIGHT, NONE])
def test_rules_classify_synthetic_poses(name):
    hf = hand_frame(name, 0, mirror_to_raw=False)
    got, conf = RuleRecognizer().classify(hf)
    assert got == name
    if name != NONE:
        assert conf > 0.5


def test_rules_left_hand_h_left_is_still_left():
    hf = hand_frame(H_LEFT, 0, hand=Hand.LEFT, mirror_to_raw=False)
    assert RuleRecognizer().classify(hf)[0] == H_LEFT


def test_smoother_emits_once_per_window_and_detects_motion():
    sm = Smoother(window_ns=250_000_000, still_px=12.0, frame_width_px=640)
    tokens = []
    for i in range(30):  # 1 s at 30 fps, still
        hf = hand_frame(OPEN_PALM, i * 33_333_333, mirror_to_raw=False)
        tok = sm.push(hf, OPEN_PALM, 0.9)
        if tok:
            tokens.append(tok)
    assert 3 <= len(tokens) <= 4
    assert all(t.name == OPEN_PALM and t.still for t in tokens)

    sm.reset()
    moving = []
    for i in range(30):  # drifting 2 % of the frame per frame
        hf = hand_frame(OPEN_PALM, i * 33_333_333, center=(0.3 + i * 0.02, 0.5), mirror_to_raw=False)
        tok = sm.push(hf, OPEN_PALM, 0.9)
        if tok:
            moving.append(tok)
    assert moving and not any(t.still for t in moving)
