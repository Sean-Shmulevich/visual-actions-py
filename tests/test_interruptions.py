"""Interruption policy: a hand off-screen keeps an armed or repeat window alive until its
own deadline instead of ending it at escape_lost; the fist stays the cancel."""

from pathlib import Path

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import default_config
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, IDLE, REPEAT, ModeEngine, Timing
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token
from visual_actions.tools.replay import replay_full
from visual_actions.tools.synth import write_session

S = 1_000_000_000
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))
NEXT = Action(ActionKind.KEY, "Next tab", (("chord", "cmd+shift+]"), ("repeat", "true")))


def make(**timing):
    bus = Bus()
    fired, modes = [], []
    bus.subscribe(ModeChanged, modes.append)
    kw = {"leader_hold_ns": 1 * S, "confidence_gain": 1.0, "escape_lost_ns": int(1.5 * S), "command_timeout_ns": 5 * S}
    kw.update(timing)
    eng = ModeEngine(
        bus,
        Bindings([Binding("window", "h_left", CMD_TAB), Binding("window", "two_up", NEXT)]),
        Timing(**kw),
        fire=lambda a, t: fired.append((a.name, t)),
    )
    return eng, fired, modes


def tok(name, t, conf=1.0, still=True, x=0.5):
    return Token(int(t * S), name, conf, Hand.RIGHT, still, x=x, y=0.5)


def arm(eng):
    for i in range(5):
        eng.on_token(tok("open_palm", i * 0.25))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED  # deadline 6.05


# -- (a) ARMED and REPEAT keep their own deadline ---------------------------------------


def test_armed_window_outlives_escape_lost_and_the_returning_gesture_fires():
    eng, fired, _ = make()
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(3.0 * S))  # gone 1.8 s: past escape_lost, inside the window
    assert eng.state == ARMED
    eng.on_token(tok("h_left", 3.2))
    assert fired == [("Cmd+Tab", int(3.2 * S))] and eng.state == ARMED


def test_clearly_gone_with_nothing_pending_drops_the_window():
    eng, fired, _ = make()
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(4.1 * S))
    assert eng.state == ARMED  # 2.9 s
    eng.on_tick(int(4.3 * S))
    assert eng.state == IDLE and not fired  # 3.1 s = past 2 x escape_lost


def test_a_half_recognized_gesture_keeps_the_window_to_its_deadline():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("h_left", 1.1, conf=0.5))  # fire mass 0.5: not enough, pending
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(5.9 * S))
    assert eng.state == ARMED and not fired  # 4.7 s gone, pending, deadline 6.05 not yet
    eng.on_tick(int(6.1 * S))
    assert eng.state == IDLE


def test_keep_armed_off_restores_the_escape_lost_drop():
    eng, _, _ = make(keep_armed_on_lost=False)
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(2.6 * S))
    assert eng.state == ARMED
    eng.on_tick(int(2.8 * S))
    assert eng.state == IDLE


def test_returning_leader_renews_the_kept_window_once():
    eng, fired, modes = make()
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_token(tok("open_palm", 2.5))  # back, showing the palm: the user wants the menu
    assert eng.state == ARMED and modes[-1].deadline_ns == int(7.5 * S)
    eng.on_tick(int(6.2 * S))
    assert eng.state == ARMED  # the original 6.05 deadline no longer applies
    eng.on_token(tok("open_palm", 6.5))  # a palm held on does not renew again
    assert modes[-1].deadline_ns == int(7.5 * S)
    eng.on_token(tok("h_left", 7.0))
    assert fired == [("Cmd+Tab", int(7.0 * S))]
    eng.on_tick(int(12.1 * S))  # the fire chained a fresh window (12.0); it times out normally
    assert eng.state == IDLE


def test_repeat_window_survives_a_loss_and_chains_into_a_kept_armed_window():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("two_up", 1.2))
    assert eng.state == REPEAT and len(fired) == 1
    eng.on_hand_lost(int(1.4 * S))
    eng.on_tick(int(2.6 * S))
    assert eng.state == REPEAT  # the slide window (deadline 2.7) is pending
    eng.on_tick(int(2.8 * S))
    assert eng.state == ARMED  # window over: back to the menu, still without a hand
    eng.on_token(tok("h_left", 3.5))  # gone 2.1 s in all: still armed
    assert len(fired) == 2 and eng.state == ARMED


def test_repeat_then_clearly_gone_drops():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("two_up", 1.2))
    eng.on_hand_lost(int(1.4 * S))
    eng.on_tick(int(2.8 * S))
    assert eng.state == ARMED
    eng.on_tick(int(4.5 * S))  # 3.1 s gone, nothing pending
    assert eng.state == IDLE and len(fired) == 1


# -- replay level, one per behaviour --------------------------------------------------------


def test_replay_armed_window_survives_two_seconds_without_a_hand(tmp_path: Path):
    p = tmp_path / "armed_gap.jsonl"
    write_session(p, [("open_palm", 1.5), ("h_left", 1.0), ("lost", 2.0), ("h_right", 0.6), ("lost", 0.5)])
    r = replay_full(p, default_config())
    assert [f.action.name for f in r.fired] == ["Cmd+Tab", "Cmd+Shift+Tab"]
    assert "idle" not in [m.new for m in r.modes]  # armed throughout the 2 s gap
    cfg = default_config()
    cfg.timing.keep_armed_on_lost = False
    assert [f.action.name for f in replay_full(p, cfg).fired] == ["Cmd+Tab"]  # the old escape_lost drop
