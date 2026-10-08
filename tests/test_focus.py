from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.core.config import default_config
from visual_actions.core.drag import DragController, FakeWindows
from visual_actions.core.events import Bus
from visual_actions.core.pinch import PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.platform.macos.windows import focus_first_focusable
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.replay import replay_full
from visual_actions.tools.synth import write_drag_session


def pointer():
    return SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), min_cutoff=1e9, beta=0.0)


def start(x=0.5, y=0.4):
    return PinchEvent(0, PinchPhase.START, x, y, 0.12, 0.2)


def test_grab_focuses_the_window_under_the_pinch():
    wins = FakeWindows([("Front", 300, 200, 400, 300)])
    c = DragController(Bus(), wins, pointer(), focus_on_grab=True)
    assert c.on_pinch(start())
    assert wins.focused == [("Front", 500.0, 400.0)]


def test_focus_can_be_disabled():
    wins = FakeWindows([("Front", 300, 200, 400, 300)])
    c = DragController(Bus(), wins, pointer(), focus_on_grab=False)
    assert c.on_pinch(start())
    assert wins.focused == []


def test_miss_does_not_focus():
    wins = FakeWindows([("Tiny", 0, 0, 10, 10)])
    c = DragController(Bus(), wins, pointer(), focus_on_grab=True)
    assert not c.on_pinch(start())
    assert wins.focused == []


def test_pipeline_calls_focus_at_once_per_grab(tmp_path):
    p = tmp_path / "drag.jsonl"
    write_drag_session(p, palm_seconds=2.5, drag_path=[(0.5, 0.5), (0.6, 0.55)], drag_seconds=0.8)
    cfg = default_config()
    d = cfg.drag
    d.box_x0, d.box_x1, d.box_y0, d.box_y1 = 0, 1, 0, 1
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    r = replay_full(p, cfg, automation=mock)
    assert sum(1 for call in r.automation.calls if call[0] == "focus_at") == 1
    cfg.drag.focus_on_grab = False
    mock2 = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    replay_full(p, cfg, automation=mock2)
    assert not any(call[0] == "focus_at" for call in mock2.calls)


def test_focus_walk_picks_first_focusable_ancestor():
    # leaf -> pane -> window; only the pane accepts focus
    tree = {"leaf": "pane", "pane": "window", "window": None}
    focusable = {"pane"}
    focused = []
    ok = focus_first_focusable("leaf", lambda e: tree[e], lambda e: e in focusable, lambda e: focused.append(e) or True)
    assert ok and focused == ["pane"]


def test_focus_walk_gives_up_when_nothing_is_focusable():
    tree = {"leaf": "pane", "pane": None}
    assert not focus_first_focusable("leaf", lambda e: tree[e], lambda e: False, lambda e: True)


def test_focus_walk_tries_the_next_ancestor_when_setting_fails():
    tree = {"leaf": "pane", "pane": "window", "window": None}
    attempted = []

    def set_focused(e):
        attempted.append(e)
        return e == "window"

    ok = focus_first_focusable("leaf", lambda e: tree[e], lambda e: True, set_focused)
    assert ok and attempted == ["leaf", "pane", "window"]
