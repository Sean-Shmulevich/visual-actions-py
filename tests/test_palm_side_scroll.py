"""The scroll hand: an edge-on palm pointing sideways, moved up / down, scrolls the active window."""

from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.core.config import default_config
from visual_actions.core.dispatcher import Dispatcher
from visual_actions.core.events import Bus, HandSeen, ModeChanged, Tick
from visual_actions.core.modes import ARMED, SCROLL
from visual_actions.core.normalize import to_user_frame
from visual_actions.core.pipeline import Pipeline
from visual_actions.core.recognizer import OPEN_PALM, PALM_SIDE, RuleRecognizer
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.synth import hand_frame

S = 1_000_000_000


def test_rule_tells_the_edge_on_hand_from_the_open_palm():
    r = RuleRecognizer()
    side = to_user_frame(hand_frame("palm_side", 0), True)
    palm = to_user_frame(hand_frame("open_palm", 0), True)
    name, conf = r.classify(side)
    assert name == PALM_SIDE and conf >= 0.5
    assert r.classify(palm)[0] == OPEN_PALM


def test_scroll_hand_never_arms_the_menu_and_stick_scrolls_once_armed():
    cfg = default_config()
    bus = Bus()
    modes = []
    bus.subscribe(ModeChanged, modes.append)
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    pipe = Pipeline(bus, cfg, Dispatcher(bus, mock))
    t = 0.0
    for _ in range(60):  # two seconds of the scroll hand alone: not a leader
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == "idle"
    for _ in range(75):  # arm with the open palm
        bus.publish(HandSeen(hand_frame("open_palm", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == ARMED
    for _ in range(15):  # the scroll hand arrives and holds still: the scroll starts here
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == SCROLL and pipe.scroll.active
    for _ in range(45):  # held 0.15 above the anchor for 1.5 s: continuous scrolling up
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.35))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    ups = [c for c in mock.calls if c[0] == "scroll"]
    assert len(ups) >= 10 and all(c[2] > 0 for c in ups)
    before = len(mock.calls)
    for _ in range(30):  # back at the anchor: nothing
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert len(mock.calls) == before
    for _ in range(30):  # below: scrolling down
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.7))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert any(c[0] == "scroll" and c[2] < 0 for c in mock.calls[before:])
    for _ in range(45):  # the shape changes for 1.5 s: scroll ends, menu stays open
        bus.publish(HandSeen(hand_frame("open_palm", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == ARMED and not pipe.scroll.active


def test_stick_velocity_curve():
    from visual_actions.core.pointer import PointerMap, ReachBox
    from visual_actions.core.scroll import ScrollController

    sent = []
    c = ScrollController(Bus(), lambda dx, dy: sent.append(dy), PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), deadzone=0.03, span=0.25, max_lines_s=90, curve=1.5)
    assert c.velocity_for(0.0) == 0.0 and c.velocity_for(0.02) == 0.0 and c.velocity_for(-0.02) == 0.0
    assert c.velocity_for(-0.25) == 90.0 and c.velocity_for(0.25) == -90.0  # hand up = scroll up
    assert 0 < c.velocity_for(-0.1) < c.velocity_for(-0.2) < 90
    c.start(0, 0.5, 0.5)
    for k in range(1, 31):
        c.update(int(k * S / 30), 0.5, 0.25)  # full speed for one second
    assert 80 <= sum(sent) <= 90
    c.end(S)
    assert not c.active and c.total_lines == sum(sent)


def test_moving_scroll_hand_shows_where_the_anchor_will_be_and_a_brief_loss_keeps_the_scroll():
    from visual_actions.core.events import HandLost
    from visual_actions.core.scroll import ScrollEvent, ScrollPhase

    cfg = default_config()
    bus = Bus()
    events = []
    bus.subscribe(ScrollEvent, events.append)
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    pipe = Pipeline(bus, cfg, Dispatcher(bus, mock))
    t = 0.0
    for _ in range(75):
        bus.publish(HandSeen(hand_frame("open_palm", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == ARMED
    for k in range(20):  # the scroll hand drifting: not still yet
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5 + 0.01 * k, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert any(e.phase is ScrollPhase.SETTLE for e in events) and pipe.engine.state == ARMED
    for _ in range(15):
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.7, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == SCROLL
    bus.publish(HandLost(int(t * S), "edge", "fingers out"))
    bus.publish(Tick(int((t + 0.5) * S)))
    assert pipe.engine.state == SCROLL  # a half-second loss keeps the anchor
    t += 0.6
    for _ in range(15):
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.7, 0.35))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == SCROLL and any(c[0] == "scroll" for c in mock.calls)
    bus.publish(HandLost(int(t * S), "edge", "gone"))
    bus.publish(Tick(int((t + 2.3) * S)))
    assert pipe.engine.state == ARMED  # gone for good: the scroll ended, the menu stays
