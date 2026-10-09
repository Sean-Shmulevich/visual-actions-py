"""Interruption policy: a hand off-screen or a tracker blip suspends an interaction instead
of ending it. Armed and repeat windows keep their own deadline; the smoother and pinch
detector keep their state across a blip; a FAST hold pauses, a STRICT hold resets."""

from pathlib import Path

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import FAST, STRICT, _base_config, apply_profile, default_config
from visual_actions.core.dispatcher import Dispatcher
from visual_actions.core.drag import DragPhase
from visual_actions.core.events import Bus, HandLost, HandSeen, ModeChanged, TokenEmitted
from visual_actions.core.modes import ARMED, HOLDING, IDLE, REPEAT, ModeEngine, Timing
from visual_actions.core.pipeline import Pipeline
from visual_actions.core.recorder import Recorder
from visual_actions.core.types import Action, ActionKind, Binding, Hand, Token
from visual_actions.platform.mock.automation import MockAutomation
from visual_actions.tools.replay import replay_full
from visual_actions.tools.synth import drag_frames, hand_frame, write_session

S = 1_000_000_000
CMD_TAB = Action(ActionKind.KEY, "Cmd+Tab", (("chord", "cmd+tab"),))
NEXT = Action(ActionKind.KEY, "Next tab", (("chord", "cmd+shift+]"), ("repeat", "true")))


def make(**timing):
    bus = Bus()
    fired, modes = [], []
    bus.subscribe(ModeChanged, modes.append)
    kw = {"leader_hold_ns": 1 * S, "confidence_gain": 1.0, "escape_lost_ns": int(1.5 * S), "command_timeout_ns": 5 * S}
    kw.update(timing)
    eng = ModeEngine(
        bus,
        Bindings([Binding("window", "h_left", CMD_TAB), Binding("window", "two_up", NEXT)]),
        Timing(**kw),
        fire=lambda a, t: fired.append((a.name, t)),
    )
    return eng, fired, modes


def tok(name, t, conf=1.0, still=True, x=0.5):
    return Token(int(t * S), name, conf, Hand.RIGHT, still, x=x, y=0.5)


def arm(eng):
    for i in range(5):
        eng.on_token(tok("open_palm", i * 0.25))
    eng.on_tick(int(1.05 * S))
    assert eng.state == ARMED  # deadline 6.05


# -- (a) ARMED and REPEAT keep their own deadline ---------------------------------------


def test_armed_window_outlives_escape_lost_and_the_returning_gesture_fires():
    eng, fired, _ = make()
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(3.0 * S))  # gone 1.8 s: past escape_lost, inside the window
    assert eng.state == ARMED
    eng.on_token(tok("h_left", 3.2))
    assert fired == [("Cmd+Tab", int(3.2 * S))] and eng.state == ARMED


def test_clearly_gone_with_nothing_pending_drops_the_window():
    eng, fired, _ = make()
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(4.1 * S))
    assert eng.state == ARMED  # 2.9 s
    eng.on_tick(int(4.3 * S))
    assert eng.state == IDLE and not fired  # 3.1 s = past 2 x escape_lost


def test_a_half_recognized_gesture_keeps_the_window_to_its_deadline():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("h_left", 1.1, conf=0.5))  # fire mass 0.5: not enough, pending
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(5.9 * S))
    assert eng.state == ARMED and not fired  # 4.7 s gone, pending, deadline 6.05 not yet
    eng.on_tick(int(6.1 * S))
    assert eng.state == IDLE


def test_keep_armed_off_restores_the_escape_lost_drop():
    eng, _, _ = make(keep_armed_on_lost=False)
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_tick(int(2.6 * S))
    assert eng.state == ARMED
    eng.on_tick(int(2.8 * S))
    assert eng.state == IDLE


def test_returning_leader_renews_the_kept_window_once():
    eng, fired, modes = make()
    arm(eng)
    eng.on_hand_lost(int(1.2 * S))
    eng.on_token(tok("open_palm", 2.5))  # back, showing the palm: the user wants the menu
    assert eng.state == ARMED and modes[-1].deadline_ns == int(7.5 * S)
    eng.on_tick(int(6.2 * S))
    assert eng.state == ARMED  # the original 6.05 deadline no longer applies
    eng.on_token(tok("open_palm", 6.5))  # a palm held on does not renew again
    assert modes[-1].deadline_ns == int(7.5 * S)
    eng.on_token(tok("h_left", 7.0))
    assert fired == [("Cmd+Tab", int(7.0 * S))]
    eng.on_tick(int(12.1 * S))  # the fire chained a fresh window (12.0); it times out normally
    assert eng.state == IDLE


def test_repeat_window_survives_a_loss_and_chains_into_a_kept_armed_window():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("two_up", 1.2))
    assert eng.state == REPEAT and len(fired) == 1
    eng.on_hand_lost(int(1.4 * S))
    eng.on_tick(int(2.6 * S))
    assert eng.state == REPEAT  # the slide window (deadline 2.7) is pending
    eng.on_tick(int(2.8 * S))
    assert eng.state == ARMED  # window over: back to the menu, still without a hand
    eng.on_token(tok("h_left", 3.5))  # gone 2.1 s in all: still armed
    assert len(fired) == 2 and eng.state == ARMED


def test_repeat_then_clearly_gone_drops():
    eng, fired, _ = make()
    arm(eng)
    eng.on_token(tok("two_up", 1.2))
    eng.on_hand_lost(int(1.4 * S))
    eng.on_tick(int(2.8 * S))
    assert eng.state == ARMED
    eng.on_tick(int(4.5 * S))  # 3.1 s gone, nothing pending
    assert eng.state == IDLE and len(fired) == 1


# -- (b) smoother and pinch detector keep state across a blip ---------------------------


def pipeline(cfg=None):
    cfg = cfg or default_config()
    bus = Bus()
    pipe = Pipeline(bus, cfg, Dispatcher(bus, MockAutomation()))
    tokens = []
    bus.subscribe(TokenEmitted, tokens.append)
    return bus, pipe, tokens


def feed(bus, name, t0, seconds, fps=30):
    t = t0
    for _ in range(int(seconds * fps)):
        bus.publish(HandSeen(hand_frame(name, int(t * S), mirror_to_raw=True)))
        t += 1 / fps
    return t


def test_blip_keeps_the_pinch_and_the_token_window():
    bus, pipe, tokens = pipeline()
    t = feed(bus, "pinch", 0.0, 0.5)
    assert pipe.pinch.pinched and len(pipe.smoother._buf) > 1
    bus.publish(HandLost(int(t * S)))
    assert pipe.pinch.pinched  # nothing is thrown away until the hand stays away
    n = len(tokens)
    feed(bus, "pinch", t + 0.1, 0.1)  # back after 100 ms
    assert pipe.pinch.pinched and len(pipe.smoother._buf) > 3  # history kept: no 2-frame re-form, no refill
    assert tokens[n].token.t_ns - int((t + 0.1) * S) < 0.1 * S  # a token within the first frames back


def test_a_longer_loss_resets_both():
    bus, pipe, _ = pipeline()
    t = feed(bus, "pinch", 0.0, 0.5)
    bus.publish(HandLost(int(t * S)))
    bus.publish(HandSeen(hand_frame("pinch", int((t + 0.4) * S))))
    assert not pipe.pinch.pinched and len(pipe.smoother._buf) == 1
    feed(bus, "pinch", t + 0.4 + 1 / 30, 0.1)
    assert pipe.pinch.pinched  # re-formed over the debounce


def test_blip_forgets_a_half_formed_pinch_change():
    bus, pipe, _ = pipeline()
    t = feed(bus, "pinch", 0.0, 0.5)
    bus.publish(HandSeen(hand_frame("open_palm", int(t * S))))  # one open frame: release pending
    assert pipe.pinch.pinched and pipe.pinch._pending == 1
    bus.publish(HandLost(int((t + 1 / 30) * S)))
    bus.publish(HandSeen(hand_frame("open_palm", int((t + 0.1) * S))))
    assert pipe.pinch.pinched  # the frame beside the loss does not count: a release needs release_frames fresh ones
    bus.publish(HandSeen(hand_frame("open_palm", int((t + 0.1 + 1 / 30) * S))))
    assert pipe.pinch.pinched  # two of three
    bus.publish(HandSeen(hand_frame("open_palm", int((t + 0.1 + 2 / 30) * S))))
    assert not pipe.pinch.pinched


# -- (c) HOLDING: FAST pauses for a short grace, STRICT resets --------------------------


def palms(eng, start, end):
    t = start
    while t <= end + 1e-9:
        eng.on_token(tok("open_palm", t))
        t += 0.25


def test_strict_hold_resets_the_moment_the_hand_is_lost():
    cfg = apply_profile(_base_config(), STRICT)
    assert cfg.timing.hold_lost_grace_s == 0.0
    eng = ModeEngine(Bus(), Bindings(), cfg.timing.to_timing(), fire=lambda a, t: None)
    palms(eng, 0, 0.5)
    assert eng.state == HOLDING
    eng.on_hand_lost(int(0.6 * S))
    assert eng.state == IDLE


def test_fast_hold_pauses_across_a_short_loss_and_the_gap_earns_nothing():
    assert apply_profile(_base_config(), FAST).timing.hold_lost_grace_s == 0.3
    eng, _, _ = make(hold_lost_grace_ns=int(0.3 * S))
    palms(eng, 0, 0.5)
    assert eng.hold_evidence_ns == 0.5 * S
    eng.on_hand_lost(int(0.6 * S))
    eng.on_tick(int(0.8 * S))
    assert eng.state == HOLDING and eng.projected_hold_ns(int(0.8 * S)) == 0.5 * S  # paused, not projected
    eng.on_hand_seen(int(0.85 * S))
    eng.on_tick(int(1.0 * S))  # the grace clock stopped when the hand was seen
    assert eng.state == HOLDING
    eng.on_token(tok("open_palm", 1.1))
    assert eng.hold_evidence_ns == 0.5 * S  # the 0.5 s away did not count as palm
    palms(eng, 1.35, 1.6)
    eng.on_tick(int(1.65 * S))
    assert eng.state == ARMED


def test_fast_hold_drops_after_the_grace():
    eng, _, _ = make(hold_lost_grace_ns=int(0.3 * S))
    palms(eng, 0, 0.5)
    eng.on_hand_lost(int(0.6 * S))
    eng.on_tick(int(0.85 * S))
    assert eng.state == HOLDING
    eng.on_tick(int(0.95 * S))
    assert eng.state == IDLE


# -- replay level, one per behaviour --------------------------------------------------------


def test_replay_armed_window_survives_two_seconds_without_a_hand(tmp_path: Path):
    p = tmp_path / "armed_gap.jsonl"
    write_session(p, [("open_palm", 2.5), ("h_left", 1.0), ("lost", 2.0), ("h_right", 0.6), ("lost", 0.5)])
    r = replay_full(p, default_config())
    assert [f.action.name for f in r.fired] == ["Previous tab", "Next tab"]
    assert "idle" not in [m.new for m in r.modes]  # armed throughout the 2 s gap
    cfg = default_config()
    cfg.timing.keep_armed_on_lost = False
    assert [f.action.name for f in replay_full(p, cfg).fired] == ["Previous tab"]  # the old escape_lost drop


def test_replay_blip_mid_drag_resumes_on_the_first_frame_back(tmp_path: Path):
    def session(path: Path, gap: float):
        rec = Recorder(path)
        t = 0.0
        for _ in range(75):  # palm 2.5 s
            rec.write(hand_frame("open_palm", int(t * S), center=(0.5, 0.5)))
            t += 1 / 30
        for hf in drag_frames(t, [(0.5, 0.5), (0.6, 0.5)], 0.5):
            rec.write(hf)
        t += 0.5
        rec.write_lost(int(t * S))
        t += gap
        back = t
        for hf in drag_frames(t, [(0.6, 0.5), (0.7, 0.5)], 0.5):
            rec.write(hf)
        t += 0.5
        for _ in range(12):
            rec.write(hand_frame("open_palm", int(t * S), center=(0.7, 0.5)))
            t += 1 / 30
        rec.write_lost(int(t * S))
        rec.close()
        return int(back * S)

    cfg = default_config()
    cfg.drag.box_x0, cfg.drag.box_x1, cfg.drag.box_y0, cfg.drag.box_y1 = 0, 1, 0, 1
    for gap, frames_to_resume in ((0.1, 1), (0.4, 2)):
        p = tmp_path / f"blip_{gap}.jsonl"
        back = session(p, gap)
        r = replay_full(p, cfg, automation=MockAutomation(screen=(1440, 900)))
        phases = [d.phase for d in r.drags if d.phase is not DragPhase.MOVE]
        assert phases == [DragPhase.START, DragPhase.PAUSE, DragPhase.RESUME, DragPhase.END]
        resume = next(d for d in r.drags if d.phase is DragPhase.RESUME)
        # a blip keeps the pinch: the drag resumes on the first frame back; a real loss re-forms it over 2 frames
        assert round((resume.t_ns - back) / (S / 30)) == frames_to_resume - 1


def test_replay_fast_hold_pauses_over_a_blip_and_still_arms(tmp_path: Path):
    p = tmp_path / "hold_blip.jsonl"
    write_session(p, [("open_palm", 0.7), ("lost", 0.1), ("open_palm", 0.5), ("h_left", 0.6), ("lost", 0.5)])
    cfg = apply_profile(default_config(), FAST)
    cfg.timing.quick_command = False  # the palm-then-gesture shortcut would hide what the hold does
    r = replay_full(p, cfg)
    assert [f.action.name for f in r.fired] == ["Previous tab"]
    assert [m.new for m in r.modes][:3] == ["holding", "armed", "repeat"]  # one hold across the blip, arm, a repeatable fire
    cfg.timing.hold_lost_grace_s = 0.0  # what STRICT does: the loss resets the hold, 0.5 s of palm is not a leader
    r = replay_full(p, cfg)
    assert not r.fired and [m.new for m in r.modes] == ["holding", "idle", "holding", "idle"]
