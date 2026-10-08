from visual_actions.core.config import default_config
from visual_actions.core.dispatcher import Dispatcher
from visual_actions.core.events import ActionFired, Bus
from visual_actions.core.recognizer import MIDDLE_UP, RuleRecognizer
from visual_actions.core.types import Action, ActionKind
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.synth import hand_frame


def test_middle_finger_rule():
    assert RuleRecognizer().classify(hand_frame("middle_up", 0, mirror_to_raw=False))[0] == MIDDLE_UP
    assert RuleRecognizer().classify(hand_frame("point_up", 0, mirror_to_raw=False))[0] != MIDDLE_UP


def test_binding_and_open_action():
    b = default_config().bindings()
    a = b.lookup("window", "middle_up")
    assert a is not None and a.kind is ActionKind.OPEN and "dQw4w9WgXcQ" in (a.arg("target") or "")
    bus = Bus()
    fired = []
    bus.subscribe(ActionFired, fired.append)
    mock = MockAutomation()
    Dispatcher(bus, mock).dispatch(a, 1)
    assert fired[0].ok and ("open", a.arg("target")) in mock.calls


def test_open_without_target_fails_cleanly():
    bus = Bus()
    fired = []
    bus.subscribe(ActionFired, fired.append)
    Dispatcher(bus, MockAutomation()).dispatch(Action(ActionKind.OPEN, "Nothing"), 1)
    assert not fired[0].ok
