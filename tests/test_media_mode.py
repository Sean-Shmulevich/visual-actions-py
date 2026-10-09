"""Two root modes: the open palm opens "window", the peace sign (two_up) opens "media"."""

import math

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import default_config, load_config
from visual_actions.core.dispatcher import Dispatcher
from visual_actions.core.drag import DragController, FakeWindows
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import (
    ADJUST,
    ARMED,
    DRAGGING,
    HOLDING,
    IDLE,
    ModeEngine,
    Timing,
)
from visual_actions.core.normalize import canonical
from visual_actions.core.pinch import PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.core.recognizer import (
    FINGERS,
    FIST,
    OPEN_PALM,
    THUMBS_DOWN,
    THUMBS_UP,
    TWO_UP,
    RuleRecognizer,
    _extension,
)
from visual_actions.core.types import (
    PINKY_DIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_PIP,
    RING_TIP,
    Action,
    ActionKind,
    Binding,
    Hand,
    HandFrame,
    Landmark,
    Token,
)
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.synth import hand_frame

S = 1_000_000_000
TIMING = Timing(leader_hold_ns=int(1.1 * S), confidence_gain=1.0, quick_command=True, quick_command_min_hold_ns=int(0.3 * S))  # FAST-style: quick command on


def make(with_drag=False):
    cfg = default_config()
    bus = Bus()
    fired, modes = [], []
    bus.subscribe(ModeChanged, modes.append)
    drag = None
    if with_drag:
        wins = FakeWindows([("Win", 0, 0, 1000, 1000)])
        drag = DragController(bus, wins, SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), 1e9, 0.0))
    eng = ModeEngine(bus, cfg.bindings(), TIMING, fire=lambda a, t: fired.append(a.name), drag=drag, leaders=cfg.leaders())
    return eng, fired, modes


def tok(name, t, conf=1.0, still=True, x=0.5):
    return Token(int(t * S), name, conf, Hand.RIGHT, still, x=x)


def hold(eng, name, start, end, step=0.25):
    t = start
    while t <= end + 1e-9:
        eng.on_token(tok(name, t))
        t += step


def arm_media_at(eng, t0):
    hold(eng, TWO_UP, t0, t0 + 1.25)
    eng.on_tick(int((t0 + 1.3) * S))
    assert eng.state == ARMED and eng.namespace == "media"


def arm_media(eng):
    arm_media_at(eng, 0.0)


# -- config ------------------------------------------------------------------


def test_default_config_has_two_root_modes():
    cfg = default_config()
    assert cfg.leaders() == {OPEN_PALM: "window", TWO_UP: "media"}
    b = cfg.bindings()
    assert b.lookup("media", "point_up").arg("verb") == "play_pause"
    assert b.lookup("media", "h_left").arg("verb") == "next"
    assert b.lookup("media", "h_right").arg("verb") == "prev"
    assert b.lookup("media", "pinch_right").arg("verb") == "volume_up"
    assert b.lookup("media", "pinch_left").arg("verb") == "volume_down"
    for unbound in ("open_palm", "two_up", "thumbs_up", "thumbs_down"):
        assert b.lookup("media", unbound) is None
    assert b.lookup("window", "h_left").name == "Previous tab"  # window mode unchanged


def test_config_file_without_media_still_gets_it(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[[namespaces.window.bindings]]\ngesture = "h_left"\naction = { kind = "key", name = "Custom", chord = "cmd+1" }\n')
    cfg = load_config(p)
    assert cfg.leaders() == {OPEN_PALM: "window", TWO_UP: "media"}
    assert cfg.bindings().lookup("media", "h_left").name == "Next track"


def test_config_file_can_rebind_the_media_leader(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[namespaces.media]\nleader = "h_right"\n')
    cfg = load_config(p)
    assert cfg.leaders() == {OPEN_PALM: "window", "h_right": "media"}
    assert cfg.bindings().lookup("media", "h_left").name == "Next track"  # default media bindings kept


# -- mode engine ---------------------------------------------------------------


def test_peace_hold_arms_media_and_palm_hold_arms_window():
    eng, _, modes = make()
    arm_media(eng)
    assert modes[0].new == HOLDING and modes[0].namespace == "media"
    eng.on_token(tok("fist", 1.5))
    assert eng.state == IDLE
    hold(eng, OPEN_PALM, 2.0, 3.25)
    eng.on_tick(int(3.3 * S))
    assert eng.state == ARMED and eng.namespace == "window"


def test_h_signs_change_tracks():
    eng, fired, _ = make()
    arm_media(eng)
    eng.on_token(tok("h_left", 1.5))
    assert fired == ["Next track"] and eng.state == ARMED
    eng.on_token(tok("h_right", 3.5, conf=0.5))  # a moderate token needs two tokens
    assert fired == ["Next track"]
    eng.on_token(tok("h_right", 3.75, conf=0.5))
    assert fired == ["Next track", "Previous track"]


def test_point_up_is_play_pause_and_does_not_repeat():
    eng, fired, _ = make()
    arm_media(eng)
    eng.on_token(tok("point_up", 1.5))
    assert fired == ["Play/Pause"] and eng.state == ARMED
    hold(eng, "point_up", 1.75, 3.0)  # held: plays/pauses once, not on every token
    assert fired == ["Play/Pause"]


def test_open_palm_in_media_mode_does_nothing():
    eng, fired, _ = make()
    arm_media(eng)
    eng.on_token(tok(OPEN_PALM, 1.5))
    assert not fired and eng.state == ARMED


def pinch(eng, t, phase, x, y=0.5):
    """x is the user frame: +x is the user's right."""
    eng.on_pinch(PinchEvent(int(t * S), phase, x, y, 0.12, 0.2))


def test_pinch_right_raises_and_left_lowers_the_volume():
    eng, fired, modes = make()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.60)
    assert eng.state == ADJUST
    pinch(eng, 1.6, PinchPhase.MOVE, 0.40)  # moving pinch: a shape change, never settles
    pinch(eng, 1.7, PinchPhase.MOVE, 0.50)
    pinch(eng, 1.9, PinchPhase.MOVE, 0.51)  # still since 1.7, but not for 0.25 s yet
    assert not fired
    pinch(eng, 2.0, PinchPhase.MOVE, 0.50)  # settled here: anchor 0.50
    pinch(eng, 2.1, PinchPhase.MOVE, 0.53)  # under one step: nothing
    assert not fired
    pinch(eng, 2.2, PinchPhase.MOVE, 0.61)  # 0.11 to the user's right: two steps up
    assert fired == ["Volume up", "Volume up"]
    pinch(eng, 2.3, PinchPhase.MOVE, 0.54)  # 0.06 back left of the anchor at 0.60: one step down
    assert fired == ["Volume up", "Volume up", "Volume down"]
    pinch(eng, 2.4, PinchPhase.END, 0.54)
    assert eng.state == ARMED  # release: back in the media menu
    assert modes[-1].old == ADJUST


def test_vertical_pinch_travel_does_not_change_the_volume():
    eng, fired, _ = make()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.5, 0.5)
    pinch(eng, 1.8, PinchPhase.MOVE, 0.5, 0.5)  # settled
    pinch(eng, 1.9, PinchPhase.MOVE, 0.51, 0.2)
    pinch(eng, 2.0, PinchPhase.MOVE, 0.49, 0.8)
    assert not fired


def test_adjust_has_no_timeout_but_a_fist_or_lost_hand_ends_it():
    eng, _, modes = make()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.5)
    eng.on_tick(int(20 * S))
    assert eng.state == ADJUST  # pinching for as long as you like
    eng.on_hand_lost(int(20.1 * S))
    eng.on_tick(int(21.7 * S))
    assert eng.state == ARMED and modes[-1].old == ADJUST  # lost past the grace: the menu stays open
    assert modes[-1].deadline_ns == int(21.7 * S) + TIMING.command_timeout_ns  # with a fresh timeout
    eng.on_token(tok("fist", 21.8))
    assert eng.state == IDLE
    arm_media_at(eng, 22.0)
    pinch(eng, 23.5, PinchPhase.START, 0.5)
    eng.on_token(tok("fist", 23.75))
    assert eng.state == IDLE


# -- adjust anchor feedback and loss grace ---------------------------------------


def make_adjust():
    from visual_actions.core.events import AdjustEvent

    cfg = default_config()
    bus = Bus()
    fired, modes, events = [], [], []
    bus.subscribe(ModeChanged, modes.append)
    bus.subscribe(AdjustEvent, events.append)
    pmap = PointerMap(1000, 1000, ReachBox(0, 1, 0, 1))
    eng = ModeEngine(bus, cfg.bindings(), TIMING, fire=lambda a, t: fired.append(a.name), leaders=cfg.leaders(), to_screen=pmap.to_screen)
    return eng, fired, modes, events


def test_adjust_publishes_settle_start_move_and_end_in_screen_points():
    from visual_actions.core.events import AdjustPhase

    eng, fired, _, events = make_adjust()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.60, 0.40)
    pinch(eng, 1.6, PinchPhase.MOVE, 0.40, 0.40)  # moving: the settle restarts
    pinch(eng, 1.7, PinchPhase.MOVE, 0.50, 0.40)
    pinch(eng, 1.9, PinchPhase.MOVE, 0.51, 0.40)
    assert [e.phase for e in events] == [AdjustPhase.SETTLE] * 4
    assert (events[0].x, events[0].y) == (600.0, 400.0) and (events[0].ax, events[0].ay) == (600.0, 400.0)  # hollow ring on the pinch
    pinch(eng, 2.0, PinchPhase.MOVE, 0.50, 0.40)  # settled: the anchor is here
    assert events[-1].phase is AdjustPhase.START and (events[-1].ax, events[-1].ay) == (500.0, 400.0)
    pinch(eng, 2.1, PinchPhase.MOVE, 0.53, 0.42)  # under one step
    assert events[-1].phase is AdjustPhase.MOVE and events[-1].steps == 0
    assert (events[-1].ax, events[-1].ay) == (500.0, 400.0) and (events[-1].x, events[-1].y) == (530.0, 420.0)
    pinch(eng, 2.2, PinchPhase.MOVE, 0.61, 0.42)  # two steps right: the anchor moves two steps along
    assert fired == ["Volume up", "Volume up"]
    assert events[-1].phase is AdjustPhase.MOVE and events[-1].steps == 2 and round(events[-1].ax) == 600
    pinch(eng, 2.3, PinchPhase.MOVE, 0.54, 0.42)
    assert events[-1].steps == -1 and round(events[-1].ax) == 550
    pinch(eng, 2.4, PinchPhase.END, 0.54, 0.42)
    assert events[-1].phase is AdjustPhase.END and events[-1].steps == 3 and eng.state == ARMED


def test_a_short_loss_keeps_the_adjust_anchor_and_the_pinch_resumes_without_settling():
    from visual_actions.core.events import AdjustPhase

    eng, fired, modes, events = make_adjust()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.50)
    pinch(eng, 1.8, PinchPhase.MOVE, 0.50)  # settled at 0.50
    assert events[-1].phase is AdjustPhase.START
    eng.on_hand_lost(int(1.9 * S))
    eng.on_tick(int(2.4 * S))
    assert eng.state == ADJUST and modes[-1].new == ADJUST  # half a second: the anchor is kept
    pinch(eng, 2.45, PinchPhase.START, 0.56)  # the pinch re-forms a step to the right: fires at once
    assert fired == ["Volume up"] and eng.state == ADJUST
    assert events[-1].phase is AdjustPhase.MOVE and events[-1].steps == 1  # no SETTLE again
    pinch(eng, 2.5, PinchPhase.MOVE, 0.44)  # and left of the moved anchor (0.55): two steps down
    assert fired == ["Volume up", "Volume down", "Volume down"]
    eng.on_tick(int(4.0 * S))
    assert eng.state == ADJUST  # the hand is back: no grace clock runs


def test_a_long_loss_ends_the_adjust_to_the_armed_menu():
    from visual_actions.core.events import AdjustPhase

    eng, fired, modes, events = make_adjust()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.50)
    pinch(eng, 1.8, PinchPhase.MOVE, 0.50)
    eng.on_hand_lost(int(1.9 * S))
    eng.on_tick(int(2.8 * S))
    assert eng.state == ADJUST
    eng.on_tick(int(3.1 * S))  # 1.2 s gone
    assert eng.state == ARMED and eng.namespace == "media"
    assert events[-1].phase is AdjustPhase.END and events[-1].steps == 0
    assert modes[-1].old == ADJUST and modes[-1].new == ARMED
    assert modes[-1].deadline_ns == int(3.1 * S) + TIMING.command_timeout_ns
    assert not fired


def test_a_fist_during_adjust_cancels_to_idle_and_ends_the_overlay():
    from visual_actions.core.events import AdjustPhase

    eng, _, _, events = make_adjust()
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.50)
    pinch(eng, 1.8, PinchPhase.MOVE, 0.50)
    eng.on_token(tok("fist", 1.9))
    assert eng.state == IDLE and events[-1].phase is AdjustPhase.END
    eng.on_hand_lost(int(2.0 * S))  # a loss after the fist: nothing to resume
    pinch(eng, 2.5, PinchPhase.START, 0.56)
    assert eng.state == IDLE and events[-1].phase is AdjustPhase.END


def test_peace_then_quick_pinch_adjusts_at_once():
    eng, fired, _ = make()
    hold(eng, TWO_UP, 0.0, 0.5)
    pinch(eng, 0.6, PinchPhase.START, 0.5)
    assert eng.state == ADJUST and eng.namespace == "media"
    pinch(eng, 0.9, PinchPhase.MOVE, 0.5)  # held still: settled
    pinch(eng, 1.0, PinchPhase.MOVE, 0.56)
    assert fired == ["Volume up"]


def test_pinch_in_media_never_grabs_a_window():
    eng, _, _ = make(with_drag=True)
    arm_media(eng)
    pinch(eng, 1.5, PinchPhase.START, 0.5)
    assert eng.state == ADJUST and eng.drag.handle is None


def test_pinch_in_window_mode_still_drags():
    eng, _, _ = make(with_drag=True)
    hold(eng, OPEN_PALM, 0.0, 1.25)
    eng.on_tick(int(1.3 * S))
    pinch(eng, 1.5, PinchPhase.START, 0.5)
    assert eng.state == DRAGGING


def test_quick_command_from_a_short_peace():
    eng, fired, _ = make()
    hold(eng, TWO_UP, 0.0, 0.5)
    eng.on_token(tok("h_left", 0.75, conf=0.95))
    assert fired == ["Next track"]


def test_palm_then_peace_mid_hold_switches_to_media():
    eng, _, modes = make()
    eng.on_token(tok(OPEN_PALM, 0.0))  # too short for a quick command
    hold(eng, TWO_UP, 0.25, 1.5)
    eng.on_tick(int(1.55 * S))
    assert eng.state == ARMED and eng.namespace == "media"
    assert [m.namespace for m in modes if m.new == HOLDING] == ["window", "media"]


def test_short_peace_then_palm_switches_to_window():
    eng, fired, _ = make()
    hold(eng, TWO_UP, 0.0, 0.75)
    eng.on_token(tok(OPEN_PALM, 1.0))
    assert not fired and eng.state == HOLDING and eng.namespace == "window"


def test_peace_held_in_the_window_menu_does_not_open_media():
    """two_up is unbound in window mode; holding it there is not a media leader, and keeping the
    shape up after the menu times out is still not one until the hand changes shape."""
    eng, fired, _ = make()
    hold(eng, OPEN_PALM, 0.0, 1.25)
    eng.on_tick(int(1.3 * S))
    assert eng.state == ARMED and eng.namespace == "window"
    hold(eng, TWO_UP, 1.5, 6.5)  # held the whole window: unbound here, never a media hold
    eng.on_tick(int(6.55 * S))
    assert eng.state == IDLE and fired == []
    hold(eng, TWO_UP, 6.75, 8.0)  # same shape still up after the timeout: blocked
    eng.on_tick(int(8.05 * S))
    assert eng.state == IDLE
    eng.on_token(tok("none", 8.25))  # hand changes shape: a peace hold may start again
    hold(eng, TWO_UP, 8.5, 9.75)
    eng.on_tick(int(9.8 * S))
    assert eng.state == ARMED and eng.namespace == "media"


def test_media_timeout_with_peace_still_up_does_not_rearm():
    eng, _, _ = make()
    arm_media(eng)
    hold(eng, TWO_UP, 1.5, 9.0)
    eng.on_tick(int(9.05 * S))
    assert eng.state == IDLE


def test_hand_loss_clears_the_block():
    eng, _, _ = make()
    arm_media(eng)
    hold(eng, TWO_UP, 1.5, 6.5)
    eng.on_tick(int(6.55 * S))
    assert eng.state == IDLE
    eng.on_hand_lost(int(6.6 * S))
    arm_media_at(eng, 7.0)


def test_a_bound_leader_fires_only_after_release():
    """If a config binds the leader shape inside its own mode, holding it from arming never fires."""
    cfg = default_config()
    b = cfg.bindings()
    b.add(Binding("media", TWO_UP, Action(ActionKind.MEDIA, "Mute", (("verb", "mute"),))))
    fired = []
    eng = ModeEngine(Bus(), b, TIMING, fire=lambda a, t: fired.append(a.name), leaders=cfg.leaders())
    arm_media(eng)
    hold(eng, TWO_UP, 1.5, 3.0)
    assert not fired
    for t in (3.25, 3.5):  # two misreads while moving are not a release
        eng.on_token(tok("none", t))
    eng.on_token(tok(TWO_UP, 3.75))
    assert not fired
    for t in (4.0, 4.25, 4.5):
        eng.on_token(tok("none", t))
    eng.on_token(tok(TWO_UP, 4.75))
    assert fired == ["Mute"]


def test_single_leader_engine_keeps_old_behavior():
    eng = ModeEngine(Bus(), Bindings(), TIMING, fire=lambda a, t: None)
    hold(eng, TWO_UP, 0.0, 2.0)
    eng.on_tick(int(2.05 * S))
    assert eng.state == IDLE


# -- dispatcher and recognizer -------------------------------------------------


def test_media_steps_press_the_key_that_many_times():
    auto = MockAutomation()
    Dispatcher(Bus(), auto).dispatch(Action(ActionKind.MEDIA, "Volume up", (("verb", "volume_up"), ("steps", "2"))), 0)
    assert [c[0] for c in auto.calls] == ["media", "media"]


def _half_curl(hf: HandFrame, pip: int, dip: int, tip: int) -> HandFrame:
    """Bend a finger at the PIP so its straightness ratio is about 0.5, the way a thumb
    pins ring and pinky in a real peace sign."""
    lm = list(hf.landmarks)
    p = lm[pip]
    mcp = lm[pip - 1]
    seg = math.hypot(p.x - mcp.x, p.y - mcp.y)
    ang = math.atan2(p.y - mcp.y, p.x - mcp.x) + math.radians(115)
    lm[dip] = Landmark(p.x + math.cos(ang) * seg * 0.5, p.y + math.sin(ang) * seg * 0.5)
    lm[tip] = Landmark(p.x + math.cos(ang) * seg, p.y + math.sin(ang) * seg)
    return HandFrame(hf.t_ns, hf.hand, tuple(lm), hf.confidence)


def test_peace_with_half_curled_ring_and_pinky_reads_confidently():
    hf = hand_frame(TWO_UP, 0, mirror_to_raw=False)
    hf = _half_curl(_half_curl(hf, RING_PIP, RING_DIP, RING_TIP), PINKY_PIP, PINKY_DIP, PINKY_TIP)
    c = canonical(hf)
    ring, pinky = (_extension(c, *f) for f in FINGERS[2:])
    assert 0.45 < max(ring, pinky) < 0.6  # the old rule (<= 0.45) rejected this hand
    name, conf = RuleRecognizer().classify(hf)
    assert name == TWO_UP and conf >= 0.8


def test_thumbs_up_and_down_are_not_a_fist():
    rr = RuleRecognizer()
    for name in (THUMBS_UP, THUMBS_DOWN):
        got, conf = rr.classify(hand_frame(name, 0, mirror_to_raw=False))
        assert got == name and conf >= 0.75
    assert rr.classify(hand_frame(FIST, 0, mirror_to_raw=False))[0] == FIST  # tucked thumb stays a fist


def test_thumbs_up_needs_the_wrist_in_frame():
    """A hand half out through the bottom edge has its fingers guessed as folded."""
    hf = hand_frame(THUMBS_UP, 0, center=(0.5, 0.97), mirror_to_raw=False)
    assert RuleRecognizer().classify(hf)[0] != THUMBS_UP


def test_volume_actions_move_the_volume_the_overlay_reads():
    auto = MockAutomation()
    d = Dispatcher(Bus(), auto)
    b = default_config().bindings()
    start = auto.volume()[0]
    d.dispatch(b.lookup("media", "pinch_right"), 0)
    d.dispatch(b.lookup("media", "pinch_right"), 1)
    assert auto.volume() == (start + 2 / 16, False)
    d.dispatch(b.lookup("media", "pinch_left"), 2)
    assert auto.volume() == (start + 1 / 16, False)
