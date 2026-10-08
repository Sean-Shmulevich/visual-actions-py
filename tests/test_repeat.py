"""Slide-to-repeat: after a repeatable action, the same shape slid sideways fires again."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import default_config
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, REPEAT, ModeEngine, Timing
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token

S = 1_000_000_000
NEXT = Action(ActionKind.KEY, "Next tab", (("chord", "cmd+shift+]"), ("repeat", "true")))
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))


def make():
    bus = Bus()
    fired, modes = [], []
    bus.subscribe(ModeChanged, modes.append)
    eng = ModeEngine(
        bus,
        Bindings([Binding("window", "two_up", NEXT), Binding("window", "h_left", CMD_TAB)]),
        Timing(leader_hold_ns=1 * S, confidence_gain=1.0, repeat_window_ns=int(1.5 * S), repeat_slide=0.10),
        fire=lambda a, t: fired.append((a.name, t)),
    )
    for i in range(5):
        eng.on_token(Token(i * 250_000_000, "open_palm", 1.0, Hand.RIGHT, True))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED
    return eng, fired, modes


def tok(name, t, x, conf=1.0, still=True):
    return Token(t, name, conf, Hand.RIGHT, still, x=x, y=0.5)


def test_repeatable_action_enters_repeat_and_slides_fire_again():
    eng, fired, modes = make()
    eng.on_token(tok("two_up", int(1.2 * S), 0.50))
    assert fired == [("Next tab", int(1.2 * S))] and eng.state == REPEAT
    eng.on_token(tok("two_up", int(1.4 * S), 0.53))  # 0.03: not a slide
    assert len(fired) == 1
    eng.on_token(tok("two_up", int(1.6 * S), 0.64, still=False))  # 0.11 from the anchor: slide
    assert len(fired) == 2 and eng.repeat_count == 2
    eng.on_token(tok("two_up", int(1.8 * S), 0.76, still=False))  # another slide
    eng.on_token(tok("two_up", int(2.0 * S), 0.64, still=False))  # slide back the other way also counts
    assert len(fired) == 4 and eng.state == REPEAT


def test_slow_drift_while_still_never_repeats():
    eng, fired, modes = make()
    eng.on_token(tok("two_up", int(1.2 * S), 0.50))
    for i in range(1, 8):  # creeps 0.03 per token but reads as still: re-anchored each time
        eng.on_token(tok("two_up", int((1.2 + 0.2 * i) * S), 0.50 + 0.03 * i))
        eng.on_tick(int((1.2 + 0.2 * i) * S))
    assert len(fired) == 1 and eng.state == REPEAT


def test_repeat_window_times_out_to_idle_and_refreshes_on_each_repeat():
    eng, fired, modes = make()
    eng.on_token(tok("two_up", int(1.2 * S), 0.50))
    eng.on_tick(int(2.6 * S))
    assert eng.state == REPEAT  # 1.4 s
    eng.on_token(tok("two_up", int(2.65 * S), 0.62, still=False))  # repeat at 2.65 refreshes the window
    eng.on_tick(int(2.8 * S))
    assert eng.state == REPEAT
    eng.on_tick(int(4.2 * S))
    assert eng.state == ARMED and len(fired) == 2  # slide window over: back in the menu


def test_unsure_or_unbound_shapes_are_ignored_during_repeat():
    eng, fired, modes = make()
    eng.on_token(tok("two_up", int(1.2 * S), 0.50))
    eng.on_token(tok("h_left", int(1.4 * S), 0.70, conf=0.6))  # bound, but not sure enough to leave the slide
    eng.on_token(tok("open_palm", int(1.6 * S), 0.70))  # unbound
    assert len(fired) == 1 and eng.state == REPEAT


def test_a_confident_other_command_leaves_the_slide_and_fires():
    eng, fired, modes = make()
    eng.on_token(tok("two_up", int(1.2 * S), 0.50))
    eng.on_token(tok("h_left", int(1.4 * S), 0.70))
    assert [f[0] for f in fired] == ["Next tab", "Cmd+Tab"] and eng.state == ARMED


def test_non_repeatable_action_returns_to_the_menu():
    eng, fired, modes = make()
    eng.on_token(tok("h_left", int(1.2 * S), 0.50))
    assert fired[0][0] == "Cmd+Tab" and eng.state == ARMED
    eng.on_token(tok("h_left", int(1.4 * S), 0.70))
    assert len(fired) == 1


def test_default_bindings_mark_only_the_tab_actions_repeatable():
    b = default_config().bindings()
    assert b.lookup("window", "point_up").arg("repeat") == "True"
    assert b.lookup("window", "two_up").arg("repeat") == "True"
    assert b.lookup("window", "h_left").arg("repeat") is None
