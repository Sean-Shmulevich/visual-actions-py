"""Swipe detector and the ARMED + swipe path of the engine."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import default_config
from visual_actions.core.events import Bus
from visual_actions.core.modes import ARMED, IDLE, ModeEngine, Timing
from visual_actions.core.recognizer import THREE_UP, RuleRecognizer
from visual_actions.core.swipe import SwipeDetector, SwipeDirection, SwipeEvent
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token
from visual_actions.tools.synth import hand_frame

S = 1_000_000_000
MS = 1_000_000


def stroke(det, shape, x_from, x_to, frames=8, dt_ms=33, t0=0, y=0.5):
    out = []
    for i in range(frames):
        x = x_from + (x_to - x_from) * i / (frames - 1)
        hf = hand_frame("three_up", t0 + i * dt_ms * MS, center=(x, y), mirror_to_raw=False)
        ev = det.update(hf, shape)
        if ev:
            out.append(ev)
    return out


def test_three_up_rule_and_pose():
    hf = hand_frame("three_up", 0, mirror_to_raw=False)
    assert RuleRecognizer().classify(hf)[0] == THREE_UP


def test_fast_sideways_stroke_fires_once_with_direction():
    det = SwipeDetector()
    evs = stroke(det, "three_up", 0.3, 0.7)  # 0.4 of the frame in 230 ms
    assert len(evs) == 1 and evs[0].direction is SwipeDirection.RIGHT and evs[0].shape == "three_up"
    evs = stroke(det, "three_up", 0.7, 0.3, t0=2 * S)
    assert len(evs) == 1 and evs[0].direction is SwipeDirection.LEFT


def test_slow_drift_does_not_fire():
    det = SwipeDetector()
    assert stroke(det, "three_up", 0.3, 0.7, frames=40, dt_ms=50) == []  # 0.4 over 2 s: too slow


def test_vertical_motion_does_not_fire():
    det = SwipeDetector()
    out = []
    for i in range(8):
        hf = hand_frame("three_up", i * 33 * MS, center=(0.5, 0.3 + 0.05 * i), mirror_to_raw=False)
        if det.update(hf, "three_up"):
            out.append(1)
    assert not out


def test_detector_reports_the_held_shape_and_rejects_mixed_shapes():
    evs = stroke(SwipeDetector(), "open_palm", 0.3, 0.7)
    assert len(evs) == 1 and evs[0].shape == "open_palm"  # shape gating is the engine's job via bindings
    det = SwipeDetector()
    out = []
    for i in range(8):  # shape flips every frame: no majority, no swipe
        hf = hand_frame("three_up", i * 33 * MS, center=(0.3 + 0.4 * i / 7, 0.5), mirror_to_raw=False)
        if det.update(hf, "three_up" if i % 2 else "none"):
            out.append(1)
    assert not out


def test_cooldown_prevents_double_fire():
    det = SwipeDetector(cooldown_ns=450 * MS)
    a = stroke(det, "three_up", 0.2, 0.6)
    b = stroke(det, "three_up", 0.6, 0.95, t0=300 * MS)  # inside the cooldown
    assert len(a) == 1 and b == []


# -- engine -----------------------------------------------------------------------------


def make():
    bus = Bus()
    fired = []
    eng = ModeEngine(
        bus,
        Bindings(
            [
                Binding("window", "three_up_swipe_left", Action(ActionKind.KEY, "Desktop right", (("chord", "ctrl+right"),))),
                Binding("window", "three_up_swipe_right", Action(ActionKind.KEY, "Desktop left", (("chord", "ctrl+left"),))),
            ]
        ),
        Timing(leader_hold_ns=1 * S, confidence_gain=1.0),
        fire=lambda a, t: fired.append(a.name),
    )
    for i in range(5):
        eng.on_token(Token(i * 250 * MS, "open_palm", 1.0, Hand.RIGHT, True))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED
    return eng, fired


def test_armed_swipe_fires_bound_direction_and_returns_to_idle():
    eng, fired = make()
    eng.on_swipe(SwipeEvent(int(1.3 * S), SwipeDirection.LEFT, "three_up", 0.3, 2.0))
    assert fired == ["Desktop right"] and eng.state == IDLE  # not marked repeatable in this test's bindings


def test_repeatable_swipes_chain_without_a_new_palm():
    bus = Bus()
    fired = []
    rep = Action(ActionKind.KEY, "Desktop right", (("chord", "ctrl+right"), ("repeat", "true")))
    eng = ModeEngine(bus, Bindings([Binding("window", "three_up_swipe_left", rep)]), Timing(leader_hold_ns=1 * S, confidence_gain=1.0, repeat_window_ns=int(1.5 * S)), fire=lambda a, t: fired.append(a.name))
    for i in range(5):
        eng.on_token(Token(i * 250 * MS, "open_palm", 1.0, Hand.RIGHT, True))
    eng.on_tick(int(1.05 * S))
    eng.on_swipe(SwipeEvent(int(1.3 * S), SwipeDirection.LEFT, "three_up", 0.3, 2.0))
    assert eng.state == ARMED and fired == ["Desktop right"]
    eng.on_swipe(SwipeEvent(int(2.0 * S), SwipeDirection.LEFT, "three_up", 0.3, 2.0))
    assert fired == ["Desktop right"] * 2 and eng.state == ARMED
    eng.on_tick(int(3.6 * S))  # 1.6 s after the last swipe
    assert eng.state == IDLE


def test_swipe_with_unbound_shape_is_ignored():
    eng, fired = make()
    eng.on_swipe(SwipeEvent(int(1.3 * S), SwipeDirection.LEFT, "open_palm", 0.3, 2.0))
    assert not fired and eng.state == ARMED


def test_swipe_in_idle_is_ignored():
    eng, fired = make()
    eng.on_tick(int(7 * S))  # timeout
    assert eng.state == IDLE
    eng.on_swipe(SwipeEvent(int(7.1 * S), SwipeDirection.LEFT, "three_up", 0.3, 2.0))
    assert not fired


def test_default_bindings_have_three_finger_desktop_swipes():
    b = default_config().bindings()
    assert b.lookup("window", "three_up_swipe_left").arg("chord") == "ctrl+left"
    assert b.lookup("window", "three_up_swipe_right").arg("chord") == "ctrl+right"
    assert b.lookup("window", "three_up_swipe_left").arg("repeat") == "True"
