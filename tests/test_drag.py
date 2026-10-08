"""Drag controller and the DRAGGING state of the mode engine, headless."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.drag import DragController, DragEvent, DragPhase, FakeWindows
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, DRAGGING, IDLE, ModeEngine, Timing
from visual_actions.core.pinch import PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.core.types import Hand, Token

S = 1_000_000_000


def pointer(screen=(1000, 1000)):
    # identity-ish map: box 0..1 -> screen, no smoothing lag (huge cutoff)
    return SmoothedPointer(PointerMap(*screen, ReachBox(0, 1, 0, 1)), min_cutoff=1e9, beta=0.0)


def pev(phase, t, x, y, scale=0.12):
    return PinchEvent(t, phase, x, y, scale, 0.2 if phase is not PinchPhase.END else 0.8)


def make_engine():
    bus = Bus()
    wins = FakeWindows([("Front", 300, 200, 400, 300), ("Back", 0, 0, 1000, 1000)])
    drag = DragController(bus, wins, pointer(), gain=1.0)
    modes = []
    bus.subscribe(ModeChanged, modes.append)
    drags = []
    bus.subscribe(DragEvent, drags.append)
    eng = ModeEngine(bus, Bindings(), Timing(leader_hold_ns=1 * S, confidence_gain=1.0), fire=lambda a, t: None, drag=drag)
    return eng, wins, modes, drags


def arm(eng):
    for i in range(5):
        eng.on_token(Token(i * 250_000_000, "open_palm", 1.0, Hand.RIGHT, True))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED


def test_controller_moves_window_by_pointer_delta():
    bus = Bus()
    wins = FakeWindows([("Front", 300, 200, 400, 300)])
    events = []
    bus.subscribe(DragEvent, events.append)
    c = DragController(bus, wins, pointer(), gain=1.0)
    assert c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.4))  # screen (500, 400) is inside Front
    assert c.dragging and events[0].phase is DragPhase.START and events[0].window == "Front"
    c.on_pinch(pev(PinchPhase.MOVE, 33_000_000, 0.6, 0.5))  # +100, +100
    assert round(wins.windows[0]["x"]) == 400 and round(wins.windows[0]["y"]) == 300
    c.on_pinch(pev(PinchPhase.MOVE, 66_000_000, 0.4, 0.3))  # -100, -100 from grab
    assert round(wins.windows[0]["x"]) == 200 and round(wins.windows[0]["y"]) == 100
    c.on_pinch(pev(PinchPhase.END, 99_000_000, 0.4, 0.3))
    assert not c.dragging and events[-1].phase is DragPhase.END
    assert c.moves == 2


def test_gain_scales_motion():
    wins = FakeWindows([("W", 100, 100, 500, 500)])
    c = DragController(Bus(), wins, pointer(), gain=2.0)
    c.on_pinch(pev(PinchPhase.START, 0, 0.3, 0.3))
    c.on_pinch(pev(PinchPhase.MOVE, 1, 0.35, 0.3))  # +50 pointer px -> +100 window px
    assert round(wins.windows[0]["x"]) == 200


def test_pinch_over_empty_space_is_a_miss():
    bus = Bus()
    events = []
    bus.subscribe(DragEvent, events.append)
    c = DragController(bus, FakeWindows([("W", 0, 0, 100, 100)]), pointer())
    assert not c.on_pinch(pev(PinchPhase.START, 0, 0.9, 0.9))
    assert events[0].phase is DragPhase.MISS and not c.dragging
    assert not c.on_pinch(pev(PinchPhase.MOVE, 1, 0.8, 0.8))  # moves without a grab are ignored


def test_front_window_wins_z_order():
    wins = FakeWindows([("Front", 300, 200, 400, 300), ("Back", 0, 0, 1000, 1000)])
    c = DragController(Bus(), wins, pointer())
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.35))
    assert c.mover.label(c.handle) == "Front"
    c.cancel(1)
    c.on_pinch(pev(PinchPhase.START, 2, 0.1, 0.1))
    assert c.mover.label(c.handle) == "Back"


def test_engine_armed_pinch_enters_dragging_and_release_returns_idle():
    eng, wins, modes, drags = make_engine()
    arm(eng)
    eng.on_pinch(pev(PinchPhase.START, int(1.2 * S), 0.5, 0.35))
    assert eng.state == DRAGGING
    eng.on_pinch(pev(PinchPhase.MOVE, int(1.3 * S), 0.7, 0.45))
    assert round(wins.windows[0]["x"]) == 500 and round(wins.windows[0]["y"]) == 300
    # tokens are ignored while dragging, even a bound-looking one
    eng.on_token(Token(int(1.35 * S), "h_left", 1.0, Hand.RIGHT, True))
    assert eng.state == DRAGGING
    eng.on_pinch(pev(PinchPhase.END, int(1.4 * S), 0.7, 0.45))
    assert eng.state == IDLE
    assert [m.new for m in modes][-3:] == [ARMED, DRAGGING, IDLE]
    assert [d.phase for d in drags] == [DragPhase.START, DragPhase.MOVE, DragPhase.END]


def test_pinch_in_idle_is_ignored():
    eng, wins, modes, drags = make_engine()
    eng.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.35))
    eng.on_pinch(pev(PinchPhase.MOVE, 1, 0.9, 0.9))
    assert eng.state == IDLE and wins.windows[0]["x"] == 300 and not drags


def test_pinch_miss_keeps_window_armed():
    eng, wins, modes, drags = make_engine()
    wins.windows = [{"label": "Tiny", "x": 0, "y": 0, "w": 10, "h": 10}]
    arm(eng)
    eng.on_pinch(pev(PinchPhase.START, int(1.2 * S), 0.5, 0.5))
    assert eng.state == ARMED and drags[-1].phase is DragPhase.MISS


def test_hand_lost_while_dragging_releases_in_place():
    eng, wins, modes, drags = make_engine()
    arm(eng)
    eng.on_pinch(pev(PinchPhase.START, int(1.2 * S), 0.5, 0.35))
    eng.on_pinch(pev(PinchPhase.MOVE, int(1.3 * S), 0.6, 0.35))
    eng.on_hand_lost(int(1.4 * S))
    assert eng.state == IDLE and drags[-1].phase is DragPhase.END
    assert round(wins.windows[0]["x"]) == 400  # stays where it was dropped


def test_drag_times_out_nothing_the_armed_deadline_does_not_apply():
    eng, wins, modes, drags = make_engine()
    arm(eng)
    eng.on_pinch(pev(PinchPhase.START, int(1.2 * S), 0.5, 0.35))
    eng.on_tick(int(20 * S))  # long past the 5 s command window
    assert eng.state == DRAGGING


def test_engine_without_drag_controller_ignores_pinches():
    eng = ModeEngine(Bus(), Bindings(), Timing(), fire=lambda a, t: None)
    eng.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    assert eng.state == IDLE
