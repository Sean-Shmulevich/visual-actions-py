"""Hand lost mid-drag: suspend, red frame, resume without a jump, or drop after the grace."""

from pathlib import Path

from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.core.bindings import Bindings
from visual_actions.core.config import default_config
from visual_actions.core.drag import DragController, DragEvent, DragPhase, FakeWindows
from visual_actions.core.events import Bus, ModeChanged
from visual_actions.core.modes import ARMED, DRAGGING, IDLE, ModeEngine, Timing
from visual_actions.core.pinch import PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.core.recorder import Recorder
from visual_actions.core.types import Hand, Token
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.replay import replay_full
from visual_actions.tools.synth import drag_frames, hand_frame

S = 1_000_000_000


def pev(phase, t, x, y):
    return PinchEvent(t, phase, x, y, 0.12, 0.2 if phase is not PinchPhase.END else 0.8)


def make():
    bus = Bus()
    wins = FakeWindows([("Win", 300, 200, 400, 300), ("Back", 0, 0, 1000, 1000)])
    drag = DragController(bus, wins, SmoothedPointer(PointerMap(1000, 1000, ReachBox(0, 1, 0, 1)), 1e9, 0.0))
    modes, drags = [], []
    bus.subscribe(ModeChanged, modes.append)
    bus.subscribe(DragEvent, drags.append)
    eng = ModeEngine(bus, Bindings(), Timing(leader_hold_ns=1 * S, confidence_gain=1.0, drag_lost_grace_ns=2 * S), fire=lambda a, t: None, drag=drag)
    for i in range(5):
        eng.on_token(Token(i * 250_000_000, "open_palm", 1.0, Hand.RIGHT, True))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED
    eng.on_pinch(pev(PinchPhase.START, int(1.2 * S), 0.5, 0.35))  # (500, 350) inside Win
    eng.on_pinch(pev(PinchPhase.MOVE, int(1.3 * S), 0.6, 0.35))  # window now at x=400
    assert eng.state == DRAGGING and round(wins.windows[0]["x"]) == 400
    return eng, wins, modes, drags


def test_hand_lost_suspends_and_window_stays():
    eng, wins, modes, drags = make()
    eng.on_hand_lost(int(1.4 * S))
    assert eng.state == DRAGGING and eng.drag.suspended
    assert drags[-1].phase is DragPhase.PAUSE
    assert round(wins.windows[0]["x"]) == 400
    eng.on_tick(int(2.0 * S))  # inside the 2 s grace
    assert eng.state == DRAGGING


def test_hand_back_pinching_resumes_without_a_jump():
    eng, wins, modes, drags = make()
    eng.on_hand_lost(int(1.4 * S))
    # the hand reappears somewhere else entirely, already pinching
    eng.on_pinch(pev(PinchPhase.START, int(2.0 * S), 0.2, 0.8))
    assert eng.state == DRAGGING and not eng.drag.suspended
    assert drags[-1].phase is DragPhase.RESUME
    assert round(wins.windows[0]["x"]) == 400  # no jump on resume
    eng.on_pinch(pev(PinchPhase.MOVE, int(2.1 * S), 0.3, 0.8))  # +100 px relative to the new anchor
    assert round(wins.windows[0]["x"]) == 500
    eng.on_pinch(pev(PinchPhase.END, int(2.2 * S), 0.3, 0.8))
    eng.on_tick(int(2.7 * S))  # past the re-grab grace
    assert eng.state == ARMED and drags[-1].phase is DragPhase.END  # a normal release: back in the menu


def test_grace_expiry_drops_in_place():
    eng, wins, modes, drags = make()
    eng.on_hand_lost(int(1.4 * S))
    eng.on_tick(int(3.3 * S))
    assert eng.state == DRAGGING  # 1.9 s
    eng.on_tick(int(3.45 * S))
    assert eng.state == IDLE and drags[-1].phase is DragPhase.END and drags[-1].snapped is None
    assert round(wins.windows[0]["x"]) == 400


def test_hand_back_open_drops_after_the_resume_grace():
    eng, wins, modes, drags = make()
    eng.on_hand_lost(int(1.4 * S))
    eng.on_token(Token(int(1.9 * S), "none", 0.6, Hand.RIGHT, True))  # back, pinch not recognised yet
    assert eng.state == DRAGGING  # given a moment
    eng.on_token(Token(int(2.15 * S), "none", 0.6, Hand.RIGHT, True))
    assert eng.state == DRAGGING
    eng.on_token(Token(int(2.6 * S), "open_palm", 1.0, Hand.RIGHT, True))  # 0.7 s later, still no pinch: drop
    assert eng.state == IDLE and drags[-1].phase is DragPhase.END
    assert round(wins.windows[0]["x"]) == 400


def test_hand_back_then_pinch_within_grace_resumes():
    eng, wins, modes, drags = make()
    eng.on_hand_lost(int(1.4 * S))
    eng.on_token(Token(int(1.9 * S), "none", 0.6, Hand.RIGHT, True))
    eng.on_pinch(pev(PinchPhase.START, int(2.2 * S), 0.2, 0.8))
    assert eng.state == DRAGGING and not eng.drag.suspended and drags[-1].phase is DragPhase.RESUME


def test_pipeline_gap_in_the_middle_of_a_drag_resumes(tmp_path: Path):
    p = tmp_path / "gap.jsonl"
    rec = Recorder(p)
    t = 0.0
    for _ in range(75):  # palm 2.5 s: past the 1.5 s hold plus the token window
        rec.write(hand_frame("open_palm", int(t * 1e9), center=(0.5, 0.5)))
        t += 1 / 30
    for hf in drag_frames(t, [(0.5, 0.5), (0.6, 0.5)], 0.6):  # drag right 0.1
        rec.write(hf)
    t += 0.6
    rec.write_lost(int(t * 1e9))
    t += 0.5  # half a second with no hand
    for hf in drag_frames(t, [(0.3, 0.5), (0.4, 0.5)], 0.6):  # back somewhere else, drag right another 0.1
        rec.write(hf)
    t += 0.6
    for _ in range(12):
        rec.write(hand_frame("open_palm", int(t * 1e9), center=(0.4, 0.5)))
        t += 1 / 30
    rec.write_lost(int(t * 1e9))
    rec.close()

    cfg = default_config()
    d = cfg.drag
    d.box_x0, d.box_x1, d.box_y0, d.box_y1 = 0, 1, 0, 1
    d.smooth_min_cutoff, d.smooth_beta = 1e6, 0.0
    mock = MockAutomation(windows=[WindowInfo(1, 1, "App", "Big", Rect(0, 0, 1440, 900))], screen=(1440, 900))
    r = replay_full(p, cfg, automation=mock)
    phases = [x.phase for x in r.drags if x.phase is not DragPhase.MOVE]
    assert phases == [DragPhase.START, DragPhase.PAUSE, DragPhase.RESUME, DragPhase.END]
    assert [m.new for m in r.modes] == ["holding", "armed", "dragging", "armed"]
    # two 0.1-of-screen moves, no jump from the hand reappearing 0.3 further left
    assert abs(r.automation.windows[0].frame.x - 0.2 * 1440) < 40
