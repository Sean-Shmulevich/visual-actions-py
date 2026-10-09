"""One finger in the window menu: slide left / right switches desktops; the shape never fires alone."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, IDLE, ModeEngine, Timing
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token

S = 1_000_000_000
LEFT = Action(ActionKind.KEY, "Desktop left", (("chord", "ctrl+left"),))
RIGHT = Action(ActionKind.KEY, "Desktop right", (("chord", "ctrl+right"),))
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))


def tok(name, t, x, conf=0.95, still=False):
    return Token(int(t * S), name, conf, Hand.RIGHT, still, x=x, y=0.5)


def make():
    bus, fired, modes = Bus(), [], []
    bus.subscribe(ModeChanged, modes.append)
    eng = ModeEngine(
        bus,
        Bindings([Binding("window", "point_up:left", LEFT), Binding("window", "point_up:right", RIGHT), Binding("window", "h_left", CMD_TAB)]),
        Timing(leader_hold_ns=1 * S, confidence_gain=1.0, repeat_slide=0.10),
        fire=lambda a, t: fired.append(a.name),
    )
    for i in range(5):
        eng.on_token(Token(i * 250_000_000, "open_palm", 1.0, Hand.RIGHT, True, x=0.5, y=0.5))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED
    return eng, fired, modes


def test_slide_right_then_left_switches_desktops_and_the_shape_alone_never_fires():
    eng, fired, modes = make()
    eng.on_token(tok("point_up", 1.2, 0.50, still=True))  # shown: anchored, nothing fires
    eng.on_token(tok("point_up", 1.4, 0.55))  # 0.05: not a slide
    assert fired == [] and eng.state == ARMED
    eng.on_token(tok("point_up", 1.6, 0.62))  # 0.12 to the user's right
    assert fired == ["Desktop right"]
    eng.on_token(tok("point_up", 1.8, 0.74))  # chained: another step right from the new anchor
    assert fired == ["Desktop right", "Desktop right"]
    eng.on_token(tok("point_up", 2.0, 0.60))  # back left 0.14
    assert fired == ["Desktop right", "Desktop right", "Desktop left"]
    assert eng.state == ARMED and all(m.new == ARMED for m in modes[-3:])  # each step renews the menu


def test_flick_anchors_at_once_and_a_loss_mid_stroke_completes_it_without_a_return_switch():
    eng, fired = make_flick()
    eng.on_token(tok("point_up", 1.2, 0.50))  # moving on arrival: anchored here anyway
    eng.on_token(tok("point_up", 1.4, 0.44))  # heading left, under a step
    eng.on_hand_lost(int(1.5 * S))  # out of the frame mid-stroke: the flick completes
    assert fired == ["Desktop left"]
    eng.on_token(tok("point_up", 2.0, 0.40))  # back in frame, still left of the anchor
    eng.on_token(tok("point_up", 2.2, 0.49))  # the return: never a right
    assert fired == ["Desktop left"]
    eng.on_token(tok("point_up", 2.4, 0.62))  # a fresh flick right from the anchor
    assert fired == ["Desktop left", "Desktop right"]


def test_a_still_hand_re_anchors_so_drift_never_switches():
    eng, fired, _ = make()
    eng.on_token(tok("point_up", 1.2, 0.50, still=True))
    for i in range(1, 8):
        eng.on_token(tok("point_up", 1.2 + 0.2 * i, 0.50 + 0.04 * i, still=True))  # creeping 0.04 per token, still
    assert fired == []


def test_other_commands_still_fire_and_the_slide_resets_on_a_new_shape():
    eng, fired, _ = make()
    eng.on_token(tok("point_up", 1.2, 0.50, still=True))
    eng.on_token(tok("h_left", 1.4, 0.50, conf=0.95, still=True))
    assert fired == ["Cmd+Tab"]
    eng.on_token(tok("point_up", 1.7, 0.80, still=True))  # a fresh anchor far from the old one: no fire
    assert fired == ["Cmd+Tab"]
    eng.on_token(tok("point_up", 1.9, 0.68))
    assert fired == ["Cmd+Tab", "Desktop left"]


def test_fist_ends_the_menu_mid_slide():
    eng, fired, _ = make()
    eng.on_token(tok("point_up", 1.2, 0.50, still=True))
    eng.on_token(tok("fist", 1.4, 0.50, conf=0.9))
    assert eng.state == IDLE and fired == []


def test_unsure_slide_shape_is_ignored():
    eng, fired, _ = make()
    eng.on_token(tok("point_up", 1.2, 0.50, conf=0.2, still=True))
    eng.on_token(tok("point_up", 1.4, 0.70, conf=0.2))
    assert fired == []


def make_flick():
    bus, fired = Bus(), []
    flick = lambda a: Action(a.kind, a.name, a.args + (("flick", "true"),))  # noqa: E731
    eng = ModeEngine(
        bus,
        Bindings([Binding("window", "point_up:left", flick(LEFT)), Binding("window", "point_up:right", flick(RIGHT))]),
        Timing(leader_hold_ns=1 * S, confidence_gain=1.0, repeat_slide=0.10),
        fire=lambda a, t: fired.append(a.name),
    )
    for i in range(5):
        eng.on_token(Token(i * 250_000_000, "open_palm", 1.0, Hand.RIGHT, True, x=0.5, y=0.5))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED
    return eng, fired


def test_flick_out_and_back_is_one_switch_and_the_return_is_not_the_opposite():
    eng, fired = make_flick()
    eng.on_token(tok("point_up", 1.2, 0.50, still=True))  # anchored at the centre
    eng.on_token(tok("point_up", 1.4, 0.38))  # 0.12 to the left: fires
    assert fired == ["Desktop left"]
    eng.on_token(tok("point_up", 1.6, 0.30))  # further left: nothing more
    eng.on_token(tok("point_up", 1.8, 0.42))  # coming back past the fire point: not a right
    assert fired == ["Desktop left"]
    eng.on_token(tok("point_up", 2.0, 0.52))  # back at the centre
    eng.on_token(tok("point_up", 2.2, 0.64))  # a new flick to the right
    assert fired == ["Desktop left", "Desktop right"]
    eng.on_token(tok("point_up", 2.4, 0.50))  # home
    eng.on_token(tok("point_up", 2.6, 0.37))  # and left again
    assert fired == ["Desktop left", "Desktop right", "Desktop left"]


def test_flick_needs_the_return_before_a_second_switch_the_same_way():
    eng, fired = make_flick()
    eng.on_token(tok("point_up", 1.2, 0.50, still=True))
    eng.on_token(tok("point_up", 1.4, 0.38))
    eng.on_token(tok("point_up", 1.6, 0.26))  # kept going left: still one switch
    eng.on_token(tok("point_up", 1.8, 0.26, still=True))  # resting out there does not re-anchor mid-flick
    eng.on_token(tok("point_up", 2.0, 0.14))
    assert fired == ["Desktop left"]
