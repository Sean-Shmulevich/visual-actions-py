"""End to end: a recorded (synthetic) palm-then-pinch-drag session moves the mock window."""

from pathlib import Path

import pytest

from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.core.config import default_config
from visual_actions.core.drag import DragPhase
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.replay import replay_full
from visual_actions.tools.synth import write_drag_session

FIXTURES = Path(__file__).parent / "fixtures"


def cfg_full_box():
    cfg = default_config()
    d = cfg.drag
    d.box_x0, d.box_x1, d.box_y0, d.box_y1 = 0.0, 1.0, 0.0, 1.0  # frame == screen for easy arithmetic
    d.smooth_min_cutoff, d.smooth_beta = 1e6, 0.0  # no smoothing lag in tests
    return cfg


def big_window_mock():
    return MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))


@pytest.fixture
def session(tmp_path):
    p = tmp_path / "drag.jsonl"
    write_drag_session(p, palm_seconds=1.5, drag_path=[(0.5, 0.5), (0.7, 0.6)], drag_seconds=1.0)
    return p


def test_drag_moves_window_by_mapped_delta(session):
    r = replay_full(session, cfg_full_box(), automation=big_window_mock())
    phases = [d.phase for d in r.drags]
    assert phases[0] is DragPhase.START and phases[-1] is DragPhase.END
    assert [m.new for m in r.modes] == ["holding", "armed", "dragging", "idle"]
    win = r.automation.windows[0].frame
    # pointer moved +0.2 of 1440 and +0.1 of 900 (pinch point tracks the hand centre)
    assert abs(win.x - 0.2 * 1440) < 25 and abs(win.y - 0.1 * 900) < 20
    assert not r.fired  # no key action fired during the drag


def test_committed_drag_fixture():
    r = replay_full(FIXTURES / "pinch_drag.jsonl", cfg_full_box(), automation=big_window_mock())
    assert r.drags and r.drags[0].phase is DragPhase.START and r.drags[-1].phase is DragPhase.END
    assert r.automation.windows[0].frame.x > 200


def test_no_leader_no_drag(tmp_path):
    from visual_actions.core.recorder import Recorder
    from visual_actions.tools.synth import drag_frames

    p = tmp_path / "pinch_only.jsonl"
    rec = Recorder(p)
    for hf in drag_frames(0.0, [(0.5, 0.5), (0.8, 0.8)], 1.5):
        rec.write(hf)
    rec.close()
    r = replay_full(p, cfg_full_box(), automation=big_window_mock())
    assert not r.drags and r.automation.windows[0].frame.x == 0


def test_drag_disabled_in_config(session):
    cfg = cfg_full_box()
    cfg.drag.enabled = False
    r = replay_full(session, cfg, automation=big_window_mock())
    assert not r.drags and r.automation.windows[0].frame.x == 0


def test_pinch_over_no_window_is_a_miss(session):
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Corner", Rect(0, 0, 50, 50))], screen=(1440, 900))
    r = replay_full(session, cfg_full_box(), automation=mock)
    assert r.drags and all(d.phase is DragPhase.MISS for d in r.drags)
    assert r.automation.windows[0].frame.x == 0


def test_unmovable_window_still_ends_cleanly(session):
    mock = big_window_mock()
    mock.movable = False
    r = replay_full(session, cfg_full_box(), automation=mock)
    assert r.drags[-1].phase is DragPhase.END
    assert r.automation.windows[0].frame.x == 0
