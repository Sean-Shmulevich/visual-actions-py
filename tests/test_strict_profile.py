"""The app default: a clear, still palm for 1.5 s straight."""

from visual_actions.core.bindings import Bindings
from visual_actions.core.config import FAST, STRICT, TimingConfig, _base_config, apply_profile, default_config, load_config, save_config
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


def test_saved_config_keeps_only_the_profile_and_user_deltas(tmp_path):
    cfg = default_config(STRICT)
    cfg.drag.box_x0 = 0.2  # a calibrated value
    p = tmp_path / "c.toml"
    save_config(cfg, p)
    text = p.read_text()
    assert 'profile = "strict"' in text and "box_x0 = 0.2" in text
    assert "leader_hold_s" not in text and "[timing]" not in text  # defaults are not frozen into the file
    back = load_config(p)
    assert back.timing.leader_hold_s == 1.5 and back.drag.box_x0 == 0.2 and back.leader.profile == STRICT


def test_saved_fast_profile_loads_as_fast(tmp_path):
    p = tmp_path / "c.toml"
    save_config(default_config(FAST), p)
    assert "[timing]" not in p.read_text()
    back = load_config(p)
    assert back.leader.profile == FAST and back.timing.leader_hold_s == 1.1 and back.timing.quick_command is True


def test_explicit_timing_key_still_overrides_the_profile(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[leader]\nprofile = "strict"\n[timing]\nleader_hold_s = 2.0\n')
    cfg = load_config(p)
    assert cfg.timing.leader_hold_s == 2.0 and cfg.timing.quick_command is False
