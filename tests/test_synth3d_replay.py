"""The kinematic generator's sequences drive the whole pipeline on fake time."""

from pathlib import Path

import pytest

from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.core.config import FAST, default_config
from visual_actions.core.drag import DragPhase
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.replay import replay_full

FIX = Path(__file__).parent / "fixtures" / "synth3d"


def cfg():
    # the committed sequences sample a 1.5-1.8 s palm (synth3d.generate_sequence), a FAST-length hold
    c = default_config(FAST)
    d = c.drag
    d.box_x0, d.box_x1, d.box_y0, d.box_y1 = 0.0, 1.0, 0.0, 1.0
    return c


def mock():
    return MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))


@pytest.mark.parametrize("name", ["pinch_drag_0", "pinch_drag_1", "pinch_drag_2"])
def test_pinch_drag_sequences_move_the_window(name):
    r = replay_full(FIX / f"{name}.jsonl", cfg(), automation=mock())
    assert [m.new for m in r.modes] == ["holding", "armed", "dragging", "armed"]
    phases = [d.phase for d in r.drags]
    assert phases[0] is DragPhase.START and phases[-1] is DragPhase.END and phases.count(DragPhase.MOVE) >= 10
    f = r.automation.windows[0].frame
    assert (f.x, f.y) != (0, 0)
    assert not r.fired


def test_palm_hold_arms_but_never_fires_or_drags():
    r = replay_full(FIX / "palm_hold_0.jsonl", cfg(), automation=mock())
    assert "armed" in [m.new for m in r.modes]
    assert not r.drags and not r.fired


def test_none_motion_never_arms():
    r = replay_full(FIX / "none_motion_0.jsonl", cfg(), automation=mock())
    assert "armed" not in [m.new for m in r.modes]
    assert not r.drags and not r.fired
