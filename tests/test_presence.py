import pytest

from visual_actions.core.config import PresenceConfig, default_config, load_config
from visual_actions.core.presence import PresenceFilter
from visual_actions.core.types import Hand, HandFrame, Landmark
from visual_actions.tools.synth import drag_frames, hand_frame

FPS = 30.0
DT_NS = int(1e9 / FPS)


def hand(dx=0.0, dy=0.0, t_ns=0, confidence=0.9, shrink=1.0):
    hf = hand_frame("open_palm", 0, center=(0.5, 0.5), mirror_to_raw=False)
    lms = tuple(Landmark(0.5 + (lm.x - 0.5) * shrink + dx, 0.5 + (lm.y - 0.5) * shrink + dy, lm.z) for lm in hf.landmarks)
    return HandFrame(t_ns, Hand.RIGHT, lms, confidence)


def test_one_or_two_missing_frames_do_not_lose_the_hand():
    f = PresenceFilter(lost_frames=3)
    assert f.update(hand()).seen is not None
    assert f.update(None).lost is None
    assert f.update(None).lost is None
    assert f.update(hand()).seen is not None and f.present


def test_three_missing_frames_lose_the_hand_with_tracker_reason():
    f = PresenceFilter(lost_frames=3)
    f.update(hand())
    f.update(None)
    f.update(None)
    pr = f.update(None, frame_index=42)
    assert pr.lost is not None and pr.lost[0] == "tracker" and "42" in pr.lost[1]
    assert not f.present
    assert f.update(None).lost is None  # already lost: no repeat


def test_edge_frames_count_as_missing_and_carry_the_edge_reason():
    f = PresenceFilter(lost_frames=2)
    f.update(hand())
    assert f.update(hand(dx=0.6)).lost is None  # wrist outside: first missing frame
    pr = f.update(hand(dx=0.6))
    assert pr.lost is not None and pr.lost[0] == "edge" and "wrist outside" in pr.lost[1]


def test_gate_closure_is_immediate():
    f = PresenceFilter()
    f.update(hand())
    pr = f.gate_closed()
    assert pr.lost is not None and pr.lost[0] == "gate"
    assert f.gate_closed().lost is None


def test_never_seen_hand_never_reports_lost():
    f = PresenceFilter(lost_frames=1)
    assert f.update(None).lost is None and f.gate_closed().lost is None


# --- fast exits -------------------------------------------------------------


def moving(f: PresenceFilter, xs: list[float], y: float = 0.5, **kw):
    """Feed a hand centred at each x in turn, one frame apart; return the last Presence."""
    pr = None
    for i, x in enumerate(xs):
        pr = f.update(hand(dx=x - 0.5, dy=y - 0.5, t_ns=i * DT_NS, **kw))
    assert pr is not None
    return pr


def test_fast_approach_to_the_left_edge_is_lost_at_once_with_fast_exit_reason():
    f = PresenceFilter(lost_frames=3, fast_exit_speed=2.0)
    pr = moving(f, [0.5, 0.38, 0.26, 0.14])  # 3.6 fw/s; the thumb side is about to cross x = 0
    assert pr.lost is not None and pr.lost[0] == "fast-exit", pr
    assert not f.present


def test_slow_approach_keeps_the_debounce():
    f = PresenceFilter(lost_frames=3, fast_exit_speed=2.0)
    pr = moving(f, [0.5, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28])  # 0.6 fw/s
    assert pr.seen is not None and f.present
    # now gone: three missing frames, the ordinary tracker reason
    assert f.update(None).lost is None and f.update(None).lost is None
    pr = f.update(None)
    assert pr.lost is not None and pr.lost[0] == "tracker"


def test_fast_motion_far_from_any_edge_is_not_an_exit():
    f = PresenceFilter(fast_exit_speed=2.0, fast_exit_reach=0.1)
    pr = moving(f, [0.2, 0.35, 0.5, 0.65, 0.8])  # 4.5 fw/s across the middle
    assert pr.seen is not None and f.present


def test_fast_exit_can_be_switched_off():
    f = PresenceFilter(fast_exit=False)
    pr = moving(f, [0.5, 0.38, 0.26, 0.14])
    assert pr.seen is not None and f.present


def test_bottom_edge_keeps_the_wrist_exemption():
    f = PresenceFilter(fast_exit_speed=2.0)
    # racing down: the wrist leaves through the bottom first, which never counts
    pr = None
    for i, y in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
        pr = f.update(hand(dy=y - 0.5, t_ns=i * DT_NS))
    assert pr is not None and pr.seen is not None and f.present


def test_bottom_prediction_for_the_pinch_point_is_opt_in():
    ys = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # the synthetic pinch point rides 0.18 above the centre
    off = PresenceFilter(fast_exit_speed=2.0, fast_exit_bottom=False)
    on = PresenceFilter(fast_exit_speed=2.0, fast_exit_bottom=True)
    first_off = first_on = None
    for i, y in enumerate(ys):
        if first_off is None and off.update(hand(dy=y - 0.5, t_ns=i * DT_NS)).lost:
            first_off = i
        if first_on is None and on.update(hand(dy=y - 0.5, t_ns=i * DT_NS)).lost:
            first_on = i
    assert first_on is not None and (first_off is None or first_on < first_off)


def test_outside_frame_after_fast_motion_is_lost_on_that_frame():
    """A flick so fast the hand goes from well inside to outside in one frame: no debounce."""
    f = PresenceFilter(lost_frames=3, fast_exit_speed=2.0)
    moving(f, [0.5, 0.5, 0.5])
    pr = f.update(hand(dx=-0.7, t_ns=3 * DT_NS))  # 0.7 fw in one frame = 21 fw/s, wrist outside
    assert pr.lost is not None and pr.lost[0] == "fast-exit" and "left at" in pr.lost[1]


def test_tracker_dropout_after_fast_motion_toward_an_edge_is_lost_on_that_frame():
    f = PresenceFilter(lost_frames=3, fast_exit_speed=2.0)
    moving(f, [0.5, 0.41, 0.32])  # 2.7 fw/s leftward, not yet within reach of x = 0
    pr = f.update(None)
    assert pr.lost is not None and pr.lost[0] == "fast-exit" and "tracker lost" in pr.lost[1]


def test_tracker_dropout_after_slow_motion_still_debounces():
    f = PresenceFilter(lost_frames=3, fast_exit_speed=2.0)
    moving(f, [0.5, 0.49, 0.48])
    assert f.update(None).lost is None and f.update(None).lost is None
    assert f.update(None).lost is not None


def test_spread_collapse_at_the_border_is_a_phantom():
    f = PresenceFilter(fast_exit_collapse=0.5)
    moving(f, [0.5, 0.5, 0.5])
    # the tracker reports a squashed hand hugging the top edge, slowly enough that velocity alone would not trigger
    pr = f.update(hand(dx=0.0, dy=-0.45, t_ns=3 * DT_NS * 10, shrink=0.3))  # dt too long for extrapolation
    assert pr.lost is not None and pr.lost[0] == "fast-exit" and "spread collapsed" in pr.lost[1]


def test_low_confidence_at_the_border_is_a_phantom_but_not_in_the_middle():
    f = PresenceFilter(fast_exit_min_confidence=0.5)
    moving(f, [0.5, 0.5])
    assert f.update(hand(t_ns=2 * DT_NS, confidence=0.2)).seen is not None  # mid-frame: fine
    pr = f.update(hand(dx=0.42, t_ns=30 * DT_NS, confidence=0.2))  # touching the right edge, no velocity (dt too long)
    assert pr.lost is not None and "confidence" in pr.lost[1]


def test_after_a_fast_exit_a_hand_lingering_at_the_border_is_not_a_return():
    f = PresenceFilter(fast_exit_speed=2.0)
    moving(f, [0.5, 0.4, 0.3, 0.2, 0.15])
    assert not f.present
    assert f.update(hand(dx=-0.35, t_ns=10 * DT_NS)).seen is None  # still hugging the left edge
    assert not f.present
    assert f.update(hand(t_ns=11 * DT_NS)).seen is not None and f.present  # clear of the border: back


def test_from_config_and_toml_section(tmp_path):
    f = PresenceFilter.from_config(PresenceConfig(lost_frames=5, fast_exit_speed=3.5, fast_exit_bottom=True))
    assert f.lost_frames == 5 and f.fast_exit_speed == 3.5 and f.fast_exit_bottom
    assert default_config().presence == PresenceConfig()
    p = tmp_path / "config.toml"
    p.write_text("[presence]\nfast_exit = false\nfast_exit_reach = 0.2\n")
    cfg = load_config(p)
    assert cfg.presence.fast_exit is False and cfg.presence.fast_exit_reach == 0.2 and cfg.presence.lost_frames == 3


# --- synthetic stress test: fling a pinched hand out of each edge -----------------

EDGES = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}


def fling(edge: str, speed: float) -> list[HandFrame]:
    """A pinched hand from the centre straight out through `edge` at `speed` frame widths/s."""
    ex, ey = EDGES[edge]
    n = max(3, round(0.9 * FPS / speed) + 1)  # frames along the path
    dist = speed * (n - 1) / FPS  # drag_frames spends n-1 frame intervals on the path
    path = [(0.5, 0.5), (0.5 + ex * dist, 0.5 + ey * dist)]
    frames = drag_frames(1.0, path, n / FPS, fps=FPS, jitter=0.0)
    # keep flying for a few more frames so an exit is observable past the last path point
    extra = [
        hand_frame("pinch", frames[-1].t_ns + k * DT_NS, center=(0.5 + ex * (dist + k * speed / FPS), 0.5 + ey * (dist + k * speed / FPS)))
        for k in range(1, 4)
    ]
    return frames + extra


@pytest.mark.parametrize("edge", list(EDGES))
@pytest.mark.parametrize("speed", [2.0, 4.0, 8.0])
def test_flung_pinched_hand_is_lost_within_two_frames_of_its_last_in_frame_position(edge, speed):
    frames = fling(edge, speed)
    last_in = max(i for i, hf in enumerate(frames) if PresenceFilter.out_of_frame(hf) is None)
    f = PresenceFilter.from_config(PresenceConfig())
    lost_at = None
    for i, hf in enumerate(frames):
        pr = f.update(hf)
        if pr.lost is not None:
            lost_at = i
            assert pr.lost[0] == "fast-exit", pr.lost
            break
    assert lost_at is not None, f"{edge} at {speed} fw/s never declared lost"
    assert lost_at <= last_in + (4 if edge == "top" else 2), f"{edge} at {speed} fw/s: lost at frame {lost_at}, last in-frame {last_in}"
    # top: only the wrist decides (a finger pointing up lives at the top edge), so a fling out the top is caught two frames later


def test_flung_hand_with_the_debounce_only_is_late():
    """The baseline the stress test improves on: with the predictor off the loss comes 3 frames after."""
    frames = fling("left", 4.0)
    last_in = max(i for i, hf in enumerate(frames) if PresenceFilter.out_of_frame(hf) is None)
    f = PresenceFilter(fast_exit=False)
    lost_at = next(i for i, hf in enumerate(frames) if f.update(hf).lost is not None)
    assert lost_at == last_in + 3
