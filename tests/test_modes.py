"""Every transition in DESIGN.md §11 on fake time."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, HOLDING, IDLE, ModeEngine, Timing
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token

S = 1_000_000_000
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))


def make():
    bus = Bus()
    fired = []
    changes = []
    bus.subscribe(ModeChanged, changes.append)
    eng = ModeEngine(
        bus,
        Bindings([Binding("window", "h_left", CMD_TAB)]),
        Timing(leader_hold_ns=2 * S, command_timeout_ns=5 * S, escape_fist_ns=1 * S, escape_lost_ns=int(1.5 * S)),
        fire=lambda a, t: fired.append((a, t)),
    )
    return eng, fired, changes


def tok(name, t, still=True):
    return Token(t_ns=t, name=name, confidence=0.9, hand=Hand.RIGHT, still=still)


def arm(eng):
    eng.on_token(tok("open_palm", 0))
    assert eng.state == HOLDING
    eng.on_tick(1 * S)
    assert eng.state == HOLDING
    eng.on_tick(2 * S)
    assert eng.state == ARMED


def test_happy_path_fires_once_and_returns_to_idle():
    eng, fired, changes = make()
    arm(eng)
    eng.on_token(tok("h_left", int(2.5 * S)))
    assert fired == [(CMD_TAB, int(2.5 * S))]
    assert eng.state == IDLE
    assert [c.new for c in changes] == [HOLDING, ARMED, IDLE]


def test_moving_palm_does_not_hold():
    eng, fired, _ = make()
    eng.on_token(tok("open_palm", 0, still=False))
    assert eng.state == IDLE


def test_breaking_the_hold_returns_to_idle():
    eng, fired, _ = make()
    eng.on_token(tok("open_palm", 0))
    eng.on_token(tok("none", int(0.5 * S)))
    assert eng.state == IDLE
    eng.on_tick(3 * S)
    assert eng.state == IDLE and not fired


def test_timeout_without_command():
    eng, fired, changes = make()
    arm(eng)
    eng.on_tick(int(6.9 * S))
    assert eng.state == ARMED
    eng.on_tick(7 * S)
    assert eng.state == IDLE and not fired


def test_unbound_gesture_keeps_window_open():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("fist", int(2.3 * S)))
    eng.on_token(tok("none", int(2.6 * S)))
    assert eng.state == ARMED and not fired


def test_leader_palm_never_fires_in_armed():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("open_palm", int(2.5 * S)))
    assert eng.state == ARMED and not fired


def test_escape_by_fist_held_one_second():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("fist", int(2.2 * S)))
    assert eng.state == ARMED
    eng.on_token(tok("fist", int(3.3 * S)))
    assert eng.state == IDLE and not fired


def test_escape_by_hand_lost():
    eng, fired, _ = make()
    arm(eng)
    eng.on_hand_lost(int(2.2 * S))
    eng.on_tick(int(3.6 * S))
    assert eng.state == ARMED  # 1.4 s, not yet
    eng.on_tick(int(3.8 * S))
    assert eng.state == IDLE and not fired


def test_brief_hand_loss_does_not_escape():
    eng, fired, _ = make()
    arm(eng)
    eng.on_hand_lost(int(2.2 * S))
    eng.on_token(tok("h_left", int(3.0 * S)))  # hand came back inside 1.5 s
    assert fired and eng.state == IDLE
