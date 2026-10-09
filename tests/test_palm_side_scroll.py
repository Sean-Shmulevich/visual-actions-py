"""The scroll hand: an edge-on palm pointing sideways, moved up / down, scrolls the active window."""

from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.core.config import default_config
from visual_actions.core.dispatcher import Dispatcher
from visual_actions.core.events import ActionFired, Bus, HandSeen, Tick
from visual_actions.core.modes import ARMED
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


def test_edge_on_hand_never_arms_the_menu_and_scrolls_once_armed():
    cfg = default_config()
    bus = Bus()
    fired = []
    bus.subscribe(ActionFired, fired.append)
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    pipe = Pipeline(bus, cfg, Dispatcher(bus, mock))
    t = 0.0
    for _ in range(60):  # two seconds of the scroll hand alone: not a leader
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == "idle" and not fired
    for _ in range(75):  # arm with the open palm
        bus.publish(HandSeen(hand_frame("open_palm", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert pipe.engine.state == ARMED
    for _ in range(15):  # the scroll hand arrives and holds still: that is where the slide anchors
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.5))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    y = 0.5
    for k in range(60):  # then rises 0.3 of the frame over two seconds
        y = 0.5 - 0.3 * k / 59
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, y))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    ups = [f for f in fired if f.action.name == "Scroll up"]
    assert len(ups) >= 3 and all(f.ok for f in ups)
    assert [c for c in mock.calls if c[0] == "scroll"][0] == ("scroll", 0, 3)
    for k in range(60):  # and back down
        bus.publish(HandSeen(hand_frame("palm_side", int(t * S), center=(0.5, 0.2 + 0.3 * k / 59))))
        bus.publish(Tick(int(t * S)))
        t += 1 / 30
    assert any(f.action.name == "Scroll down" for f in fired)
    assert pipe.engine.state == ARMED  # scrolling keeps the menu open
