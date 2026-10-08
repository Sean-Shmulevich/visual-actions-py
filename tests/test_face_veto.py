"""Face-touch veto: a palm whose box sits inside a face box with bunched fingers is not a leader."""

from visual_actions.core.config import default_config
from visual_actions.core.dispatcher import Dispatcher
from visual_actions.core.events import Bus, HandSeen, ModeChanged, PalmVetoed, Tick
from visual_actions.core.pipeline import Pipeline, palm_vetoed
from visual_actions.core.recognizer import tip_spread
from visual_actions.core.types import Hand, HandFrame, Landmark
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.synth import hand_frame
from visual_actions.ui.face import hand_face_overlap


def bunched_palm(t_ns, mirror_to_raw=False):
    """An open palm with the finger tips pulled together (spread ~0.3), like a hand resting on a cheek."""
    hf = hand_frame("open_palm", t_ns, mirror_to_raw=mirror_to_raw)
    w = hf.landmarks[0]
    lms = []
    for i, lm in enumerate(hf.landmarks):
        if i >= 5:  # squeeze every finger landmark toward the hand's centre line
            lms.append(Landmark(w.x + (lm.x - w.x) * 0.35, lm.y, lm.z))
        else:
            lms.append(lm)
    return HandFrame(t_ns, Hand.RIGHT, tuple(lms), 0.95)


def test_spread_separates_open_palm_from_bunched():
    assert tip_spread(hand_frame("open_palm", 0, mirror_to_raw=False)) > 0.6
    assert tip_spread(bunched_palm(0)) < 0.6


def test_overlap_geometry():
    hf = hand_frame("open_palm", 0, center=(0.5, 0.5), scale=0.1, mirror_to_raw=False)
    assert hand_face_overlap(hf, []) == 0.0
    assert hand_face_overlap(hf, [(0.0, 0.0, 1.0, 1.0)]) > 0.99
    assert hand_face_overlap(hf, [(0.9, 0.9, 1.0, 1.0)]) == 0.0


def test_veto_rule():
    cfg = default_config()
    assert palm_vetoed(bunched_palm(0), 0.8, cfg)
    assert not palm_vetoed(bunched_palm(0), 0.2, cfg)  # not on a face
    assert not palm_vetoed(hand_frame("open_palm", 0, mirror_to_raw=False), 0.9, cfg)  # spread palm in front of the face
    cfg.leader.face_veto = False
    assert not palm_vetoed(bunched_palm(0), 0.8, cfg)


def run(frames_with_overlap):
    bus = Bus()
    cfg = default_config()
    cfg.recognizer.model = None
    Pipeline(bus, cfg, Dispatcher(bus, MockAutomation()))
    modes, vetoes = [], []
    bus.subscribe(ModeChanged, modes.append)
    bus.subscribe(PalmVetoed, vetoes.append)
    t = 0
    for hf, ov in frames_with_overlap:
        bus.publish(HandSeen(hf, ov))
        bus.publish(Tick(hf.t_ns))
        t = hf.t_ns
    bus.publish(Tick(t + 2_000_000_000))
    return modes, vetoes


def test_hand_on_face_never_starts_a_hold():
    frames = [(bunched_palm(i * 33_000_000, mirror_to_raw=True), 0.75) for i in range(60)]
    modes, vetoes = run(frames)
    assert not any(m.new == "holding" for m in modes)
    assert vetoes and vetoes[0].overlap == 0.75


def test_same_hand_away_from_the_face_still_arms():
    frames = [(bunched_palm(i * 33_000_000, mirror_to_raw=True), 0.0) for i in range(60)]
    modes, _ = run(frames)
    assert any(m.new == "holding" for m in modes)
