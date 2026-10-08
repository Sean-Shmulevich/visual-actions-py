"""Palm-then-gesture in one motion."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.drag import DragController, FakeWindows
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, DRAGGING, HOLDING, IDLE, ModeEngine, Timing
from visual_actions.core.pinch import PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token

S = 1_000_000_000
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))
TIMING = Timing(leader_hold_ns=int(1.1 * S), confidence_gain=1.0, quick_command_min_hold_ns=int(0.3 * S), quick_command_min_confidence=0.85)


def make(with_drag=False):
    bus = Bus()
    fired, modes = [], []
    bus.subscribe(ModeChanged, modes.append)
    drag = None
    wins = None
    if with_drag:
        wins = FakeWindows([("Win", 0, 0, 1000, 1000)])
        drag = DragController(bus, wins, SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), 1e9, 0.0))
    eng = ModeEngine(bus, Bindings([Binding("window", "h_left", CMD_TAB)]), TIMING, fire=lambda a, t: fired.append(a.name), drag=drag)
    return eng, fired, modes, wins


def palm(eng, t):
    eng.on_token(Token(int(t * S), "open_palm", 1.0, Hand.RIGHT, True))


def test_short_palm_then_confident_bound_gesture_fires_at_once():
    eng, fired, modes, _ = make()
    palm(eng, 0.0)
    palm(eng, 0.25)
    palm(eng, 0.5)  # 0.5 s of evidence, well short of the 1.1 s hold
    eng.on_token(Token(int(0.75 * S), "h_left", 0.95, Hand.RIGHT, True))
    assert fired == ["Cmd+Tab"] and eng.state == ARMED
    assert [m.new for m in modes] == [HOLDING, ARMED, ARMED]


def test_too_short_a_palm_does_not_quick_fire():
    eng, fired, modes, _ = make()
    palm(eng, 0.0)
    eng.on_token(Token(int(0.25 * S), "h_left", 0.95, Hand.RIGHT, True))  # 0.25 s < 0.3 s
    assert not fired and eng.state == HOLDING  # treated as a flicker, hold paused


def test_unbound_or_unsure_gesture_does_not_quick_fire():
    eng, fired, modes, _ = make()
    palm(eng, 0.0)
    palm(eng, 0.25)
    palm(eng, 0.5)
    eng.on_token(Token(int(0.75 * S), "two_up", 0.99, Hand.RIGHT, True))  # not bound here
    assert not fired and eng.state == HOLDING
    eng.on_token(Token(int(1.0 * S), "h_left", 0.6, Hand.RIGHT, True))  # bound but unsure: second flicker breaks
    assert not fired and eng.state == IDLE


def test_palm_then_pinch_grabs_at_once():
    eng, fired, modes, wins = make(with_drag=True)
    palm(eng, 0.0)
    palm(eng, 0.25)
    palm(eng, 0.5)
    eng.on_pinch(PinchEvent(int(0.7 * S), PinchPhase.START, 0.5, 0.5, 0.12, 0.2))
    assert eng.state == DRAGGING
    eng.on_pinch(PinchEvent(int(0.8 * S), PinchPhase.MOVE, 0.6, 0.5, 0.12, 0.2))
    assert round(wins.windows[0]["x"]) == 100


def test_pinch_too_early_in_the_hold_is_ignored():
    eng, fired, modes, wins = make(with_drag=True)
    palm(eng, 0.0)
    eng.on_pinch(PinchEvent(int(0.1 * S), PinchPhase.START, 0.5, 0.5, 0.12, 0.2))
    assert eng.state == HOLDING and wins.windows[0]["x"] == 0
