"""Pinch-drag calm: a release is final only after its re-grab grace, a re-formed pinch
continues the drag, one-frame pointer glitches are dropped, a release forming at hand loss
drops the window once the loss is clearly not a blip."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import default_config
from visual_actions.core.drag import DragController, DragPhase, FakeWindows
from visual_actions.core.events import Bus, HandLost, HandSeen, ModeChanged, Tick
from visual_actions.core.modes import ARMED, DRAGGING, ModeEngine, Timing
from visual_actions.core.pinch import PinchDetector, PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.core.types import Hand, Token
from visual_actions.tools.synth import hand_frame

S = 1_000_000_000
MS = 1_000_000


def pev(phase, t, x, y):
    return PinchEvent(t, phase, x, y, 0.12, 0.2 if phase is not PinchPhase.END else 0.8)


def controller(grace_ms=400):
    bus, events = Bus(), []
    bus.subscribe(DragEvent := __import__("visual_actions.core.drag", fromlist=["DragEvent"]).DragEvent, events.append)
    wins = FakeWindows([("Win", 0, 0, 1000, 1000)])
    pointer = SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), 1e9, 0.0, glitch_px=0)
    c = DragController(bus, wins, pointer, release_grace_ns=grace_ms * MS)
    return c, wins, events


def phases(events):
    return [e.phase for e in events if e.phase is not DragPhase.MOVE]


def test_release_regrabbed_within_grace_continues_the_same_drag():
    c, wins, events = controller()
    assert c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.MOVE, 33 * MS, 0.6, 0.5))
    assert round(wins.windows[0]["x"]) == 100
    c.on_pinch(pev(PinchPhase.END, 66 * MS, 0.6, 0.5))  # the carried pinch flickered open
    assert c.dragging and c.releasing and phases(events) == [DragPhase.START]
    c.on_tick(200 * MS)  # inside the grace: still held
    assert c.dragging
    c.on_pinch(pev(PinchPhase.START, 250 * MS, 0.62, 0.51))  # back, near the same spot
    assert c.dragging and not c.releasing and phases(events)[-1] is DragPhase.RESUME
    c.on_pinch(pev(PinchPhase.MOVE, 283 * MS, 0.72, 0.51))
    assert round(wins.windows[0]["x"]) == 200  # continued from where the window was, no jump
    c.on_pinch(pev(PinchPhase.END, 316 * MS, 0.72, 0.51))
    c.on_tick(800 * MS)
    assert not c.dragging and phases(events) == [DragPhase.START, DragPhase.RESUME, DragPhase.END]


def test_release_is_final_after_the_grace():
    c, wins, events = controller()
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.END, 100 * MS, 0.5, 0.5))
    assert not c.on_tick(400 * MS) and c.dragging
    assert c.on_tick(500 * MS) and not c.dragging
    assert events[-1].phase is DragPhase.END and events[-1].t_ns == 100 * MS  # ended at the release time


def test_regrab_far_away_or_late_is_a_new_drag():
    c, wins, events = controller()
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.END, 100 * MS, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.START, 200 * MS, 0.9, 0.9))  # 400 px away: not the same pinch
    assert phases(events) == [DragPhase.START, DragPhase.END, DragPhase.START]
    c.on_pinch(pev(PinchPhase.END, 300 * MS, 0.9, 0.9))
    c.on_pinch(pev(PinchPhase.START, 900 * MS, 0.9, 0.9))  # same spot but after the grace
    assert phases(events)[-2:] == [DragPhase.END, DragPhase.START]


def test_fist_during_the_grace_drops_in_place_without_snap():
    c, wins, events = controller()
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.END, 100 * MS, 0.5, 0.5))
    c.cancel(150 * MS)
    assert not c.dragging and events[-1].phase is DragPhase.END and events[-1].snapped is None


def test_no_grace_keeps_the_old_immediate_release():
    c, wins, events = controller(grace_ms=0)
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.END, 100 * MS, 0.5, 0.5))
    assert not c.dragging and events[-1].phase is DragPhase.END


def test_sub_pixel_moves_are_not_sent_to_the_window():
    c, wins, events = controller()
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.MOVE, 33 * MS, 0.5004, 0.5))  # 0.4 px
    assert wins.moves == []
    c.on_pinch(pev(PinchPhase.MOVE, 66 * MS, 0.503, 0.5))  # 3 px
    assert len(wins.moves) == 1


def test_engine_stays_dragging_through_the_grace_then_returns_to_the_menu():
    bus, modes = Bus(), []
    bus.subscribe(ModeChanged, modes.append)
    wins = FakeWindows([("Win", 0, 0, 1000, 1000)])
    drag = DragController(bus, wins, SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), 1e9, 0.0, glitch_px=0))
    eng = ModeEngine(bus, Bindings(), Timing(leader_hold_ns=S), fire=lambda a, t: None, drag=drag)
    for k in range(6):
        eng.on_token(Token(int(k * 0.25 * S), "open_palm", 1.0, Hand.RIGHT, True))
    eng.on_tick(int(1.6 * S))
    assert eng.state == ARMED
    eng.on_pinch(pev(PinchPhase.START, int(1.7 * S), 0.5, 0.5))
    eng.on_pinch(pev(PinchPhase.END, int(1.8 * S), 0.5, 0.5))
    eng.on_tick(int(1.9 * S))
    assert eng.state == DRAGGING  # the release may still be a flicker
    eng.on_tick(int(2.3 * S))
    assert eng.state == ARMED and [m.new for m in modes][-2:] == ["dragging", "armed"]


def test_glitch_frame_is_dropped_and_a_real_move_is_kept():
    p = SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), 1e9, 0.0, glitch_px=24.0)
    t = 0
    for k in range(6):  # still hand at (500, 500)
        out = p.update(t, 0.5, 0.5)
        t += 33 * MS
    assert out == (500.0, 500.0)
    held = p.update(t, 0.54, 0.5)  # a 40 px jump from a still hand: held back
    assert held == (500.0, 500.0)
    back = p.update(t + 33 * MS, 0.5, 0.5)  # and it came back: the jump never happened
    assert back == (500.0, 500.0)
    held = p.update(t + 66 * MS, 0.54, 0.5)  # jump again
    moved = p.update(t + 99 * MS, 0.58, 0.5)  # and kept going: a real move, both frames applied
    assert held == (500.0, 500.0) and moved[0] > 560


def test_release_forming_at_hand_loss_drops_the_window_once_it_is_not_a_blip():
    cfg = default_config()
    d = cfg.drag
    d.box_x0, d.box_x1, d.box_y0, d.box_y1 = 0, 1, 0, 1
    from visual_actions.core.dispatcher import Dispatcher
    from visual_actions.core.pipeline import Pipeline
    from visual_actions.platform.mock.automation import MockAutomation
    from visual_actions.core.automation import Rect, WindowInfo

    bus, modes = Bus(), []
    bus.subscribe(ModeChanged, modes.append)
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    pipe = Pipeline(bus, cfg, Dispatcher(bus, mock))
    t = 0.0
    for _ in range(75):
        bus.publish(HandSeen(hand_frame("open_palm", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == ARMED
    for _ in range(15):
        bus.publish(HandSeen(hand_frame("pinch", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == DRAGGING
    bus.publish(HandSeen(hand_frame("open_palm", int(t * S), center=(0.5, 0.5))))  # one open frame: release forming
    t += 1 / 30
    bus.publish(HandLost(int(t * S)))
    assert pipe.engine.state == DRAGGING and pipe.drag is not None and pipe.drag.suspended
    bus.publish(Tick(int((t + 0.1) * S)))
    assert pipe.engine.state == DRAGGING  # could still be a blip
    bus.publish(Tick(int((t + 0.2) * S)))
    assert pipe.engine.state == ARMED and not pipe.drag.dragging  # dropped in place, menu open


def test_release_needs_more_frames_than_a_grab():
    det = PinchDetector(0.3, 0.55, debounce_frames=2)
    assert det.release_frames == 3
    frames = [hand_frame("pinch", i * 33 * MS, mirror_to_raw=False) for i in range(4)]
    frames += [hand_frame("open_palm", (4 + i) * 33 * MS, mirror_to_raw=False) for i in range(2)]
    frames += [hand_frame("pinch", (6 + i) * 33 * MS, mirror_to_raw=False) for i in range(2)]
    phases_ = [e.phase for e in (det.update(f) for f in frames) if e is not None]
    assert PinchPhase.END not in phases_ and det.pinched  # two open frames are a flicker
    for i in range(3):
        det.update(hand_frame("open_palm", (8 + i) * 33 * MS, mirror_to_raw=False))
    assert not det.pinched
