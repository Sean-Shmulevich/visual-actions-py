"""Every transition in DESIGN.md §11 on fake time, with confidence-weighted timing."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.events import Bus, HoldProgress, ModeChanged
from visual_actions.core.modes import ARMED, HOLDING, IDLE, ModeEngine, Timing, hold_rate
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token

S = 1_000_000_000
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))
TIMING = Timing(leader_hold_ns=2 * S, command_timeout_ns=5 * S, escape_fist_ns=1 * S, escape_lost_ns=int(1.5 * S), confidence_gain=1.0, leader_min_confidence=0.8)


def make(timing: Timing = TIMING, **kw):
    bus = Bus()
    fired = []
    changes = []
    bus.subscribe(ModeChanged, changes.append)
    eng = ModeEngine(bus, Bindings([Binding("window", "h_left", CMD_TAB)]), timing, fire=lambda a, t: fired.append((a, t)), **kw)
    return eng, fired, changes, bus


def tok(name, t, still=True, conf=1.0):
    return Token(t_ns=t, name=name, confidence=conf, hand=Hand.RIGHT, still=still)


def palm_stream(eng, start, end, step=0.25, conf=1.0, still=True):
    t = start
    while t <= end + 1e-9:
        eng.on_token(tok("open_palm", int(t * S), still=still, conf=conf))
        t += step


def arm(eng, conf=1.0):
    palm_stream(eng, 0, 2.0, conf=conf)
    eng.on_tick(int(2.05 * S))
    assert eng.state == ARMED


def test_unsure_palm_does_not_start_or_fill_the_hold():
    eng, _, _, _ = make(Timing(leader_hold_ns=2 * S, confidence_gain=1.0, leader_min_confidence=0.8))
    eng.on_token(tok("open_palm", 0, conf=0.7))
    assert eng.state == IDLE
    palm_stream(eng, 0, 2.5, conf=0.9)  # rate 0.8: 2.5 s of wall clock is 2.0 s of evidence
    eng.on_tick(int(2.55 * S))
    assert eng.state == ARMED


def test_hold_rate_curve():
    assert hold_rate(1.0, 1.0) == 1.0
    assert hold_rate(0.75, 1.0) == 0.5
    assert hold_rate(0.5, 1.0) == 0.0
    assert hold_rate(0.0, 1.0) == -1.0
    assert hold_rate(1.0, 1.5) == 1.5


def test_happy_path_fires_once_and_returns_to_idle():
    eng, fired, changes, _ = make()
    arm(eng)
    eng.on_token(tok("h_left", int(2.5 * S)))
    assert fired == [(CMD_TAB, int(2.5 * S))]
    assert eng.state == IDLE
    assert [c.new for c in changes] == [HOLDING, ARMED, IDLE]


def test_sure_palm_arms_early_with_gain():
    eng, _, _, _ = make(Timing(leader_hold_ns=2 * S, confidence_gain=2.0))
    palm_stream(eng, 0, 1.0)  # 1 s of wall clock at rate 2.0
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED


def test_hesitant_palm_stalls_until_sure_again():
    eng, _, _, _ = make()
    palm_stream(eng, 0, 1.0)  # sure: 1 s of evidence
    palm_stream(eng, 1.25, 3.0, conf=0.75)  # under leader_min: no fill, no drain
    eng.on_tick(int(3.05 * S))
    assert eng.state == HOLDING and abs(eng.hold_evidence_ns - 1.0 * S) < 1e-6
    palm_stream(eng, 3.25, 4.25)
    eng.on_tick(int(4.3 * S))
    assert eng.state == ARMED


def test_unsure_palm_never_starts_the_hold():
    eng, _, _, _ = make()
    palm_stream(eng, 0, 10.0, conf=0.4)
    eng.on_tick(int(10.05 * S))
    assert eng.state == IDLE


def test_hold_progress_events_are_published():
    eng, _, _, bus = make()
    progress = []
    bus.subscribe(HoldProgress, progress.append)
    palm_stream(eng, 0, 1.0)
    assert progress and abs(progress[-1].fraction - 0.5) < 1e-6


def test_moving_palm_pauses_the_count():
    eng, _, _, _ = make()
    palm_stream(eng, 0, 1.0)
    palm_stream(eng, 1.25, 3.0, still=False)
    eng.on_tick(int(3.05 * S))
    assert eng.state == HOLDING
    palm_stream(eng, 3.25, 4.25)
    eng.on_tick(int(4.3 * S))
    assert eng.state == ARMED


def test_breaking_the_hold_returns_to_idle():
    eng, fired, _, _ = make()
    eng.on_token(tok("open_palm", 0))
    eng.on_token(tok("none", int(0.25 * S)))
    assert eng.state == HOLDING  # one flicker is tolerated
    eng.on_token(tok("none", int(0.5 * S)))
    assert eng.state == IDLE
    eng.on_tick(3 * S)
    assert eng.state == IDLE and not fired


def test_timeout_without_command():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_tick(int(7.0 * S))
    assert eng.state == ARMED
    eng.on_tick(int(7.1 * S))
    assert eng.state == IDLE and not fired


def test_unbound_gesture_keeps_window_open():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_token(tok("point_up", int(2.3 * S)))
    eng.on_token(tok("none", int(2.6 * S)))
    assert eng.state == ARMED and not fired


def test_leader_palm_never_fires_in_armed():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_token(tok("open_palm", int(2.5 * S)))
    assert eng.state == ARMED and not fired


def test_weak_tokens_accumulate_to_fire():
    eng, fired, _, _ = make(fire_evidence=0.9, min_token_confidence=0.3)
    arm(eng)
    eng.on_token(tok("h_left", int(2.3 * S), conf=0.5))
    assert not fired
    eng.on_token(tok("h_left", int(2.55 * S), conf=0.5))
    assert fired == [(CMD_TAB, int(2.55 * S))]


def test_too_weak_tokens_never_fire():
    eng, fired, _, _ = make(fire_evidence=0.9, min_token_confidence=0.3)
    arm(eng)
    for i in range(10):
        eng.on_token(tok("h_left", int((2.3 + i * 0.25) * S), conf=0.2))
    assert not fired and eng.state == ARMED


def test_switching_gesture_restarts_evidence():
    eng, fired, _, _ = make(fire_evidence=0.9)
    eng.bindings.add(Binding("window", "h_right", Action(ActionKind.KEY, "Cmd+Shift+Tab")))
    arm(eng)
    eng.on_token(tok("h_left", int(2.3 * S), conf=0.5))
    eng.on_token(tok("h_right", int(2.55 * S), conf=0.5))
    assert not fired
    eng.on_token(tok("h_right", int(2.8 * S), conf=0.5))
    assert fired[0][0].name == "Cmd+Shift+Tab"


def test_bound_fist_still_cancels():
    eng, fired, _, _ = make()
    eng.bindings.add(Binding("window", "fist", Action(ActionKind.KEY, "App Expos\u00e9")))
    arm(eng)
    eng.on_token(tok("fist", int(2.3 * S)))
    assert not fired and eng.state == IDLE


def test_fist_cancels_immediately_in_any_state():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_token(tok("fist", int(2.2 * S)))
    assert eng.state == IDLE and not fired
    eng.on_token(tok("open_palm", int(3.0 * S)))
    assert eng.state == HOLDING
    eng.on_token(tok("fist", int(3.25 * S), conf=0.7))
    assert eng.state == IDLE


def test_weak_fist_does_not_cancel():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_token(tok("fist", int(2.2 * S), conf=0.4))
    assert eng.state == ARMED


def test_escape_by_hand_lost():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_hand_lost(int(2.2 * S))
    eng.on_tick(int(3.6 * S))
    assert eng.state == ARMED  # 1.4 s, not yet
    eng.on_tick(int(3.8 * S))
    assert eng.state == IDLE and not fired


def test_brief_hand_loss_does_not_escape():
    eng, fired, _, _ = make()
    arm(eng)
    eng.on_hand_lost(int(2.2 * S))
    eng.on_token(tok("h_left", int(3.0 * S)))  # hand came back inside 1.5 s
    assert fired and eng.state == IDLE
