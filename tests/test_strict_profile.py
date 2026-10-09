"""The app default: a clear, still palm for 1.5 s straight."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import STRICT, TimingConfig, _base_config, apply_profile, default_config
from visual_actions.core.modes import Timing
from visual_actions.core.events import Bus
from visual_actions.core.modes import ARMED, HOLDING, IDLE, ModeEngine
from visual_actions.core.types import Hand, Token

S = 1_000_000_000


def engine():
    cfg = apply_profile(_base_config(), STRICT)
    assert cfg.timing.leader_hold_s == 1.5 and not cfg.timing.quick_command and cfg.recognizer.palm_strict
    return ModeEngine(Bus(), Bindings(), cfg.timing.to_timing(), fire=lambda a, t: None)


def palms(eng, start, end, conf=1.0, still=True):
    t = start
    while t <= end + 1e-9:
        eng.on_token(Token(int(t * S), "open_palm", conf, Hand.RIGHT, still))
        t += 0.25


def test_confidence_cannot_shorten_the_hold():
    eng = engine()
    palms(eng, 0, 1.25)
    eng.on_tick(int(1.3 * S))
    assert eng.state == HOLDING
    palms(eng, 1.5, 1.5)
    eng.on_tick(int(1.55 * S))
    assert eng.state == ARMED


def test_any_flicker_resets_the_hold():
    eng = engine()
    palms(eng, 0, 1.0)
    eng.on_token(Token(int(1.25 * S), "none", 0.6, Hand.RIGHT, True))
    assert eng.state == IDLE


def test_movement_resets_the_count():
    eng = engine()
    palms(eng, 0, 1.0)
    palms(eng, 1.25, 1.25, still=False)
    assert eng.hold_evidence_ns == 0.0 and eng.state == HOLDING
    palms(eng, 1.5, 2.5)
    eng.on_tick(int(2.55 * S))
    assert eng.state == HOLDING  # 1.25 s since the reset
    palms(eng, 2.75, 2.75)
    eng.on_tick(int(2.8 * S))
    assert eng.state == ARMED


def test_quick_command_is_off():
    eng = engine()
    palms(eng, 0, 0.5)
    eng.on_token(Token(int(0.75 * S), "h_left", 0.99, Hand.RIGHT, True))
    assert eng.state == IDLE


def test_timing_defaults_are_the_shipped_profile():
    assert Timing() == TimingConfig().to_timing() == default_config().timing.to_timing()
