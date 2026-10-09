from pathlib import Path

import pytest

from visual_actions.core.chords import parse_chord
from visual_actions.core.config import default_config, load_config, save_config
from visual_actions.core.types import ActionKind


def test_parse_chord():
    c = parse_chord("Cmd+Shift+Tab")
    assert c.modifiers == ("cmd", "shift") and c.key == "tab"
    assert parse_chord("option+left").modifiers == ("alt",)
    with pytest.raises(ValueError):
        parse_chord("cmd+")
    with pytest.raises(ValueError):
        parse_chord("foo+a")


def test_default_bindings():
    b = default_config().bindings()
    a = b.lookup("window", "h_left")
    assert a is not None and a.kind is ActionKind.KEY and a.arg("chord") == "cmd+shift+["
    assert b.lookup("window", "h_right").name == "Next tab" and b.lookup("window", "h_right").arg("repeat") == "True"
    assert b.lookup("window", "palm_side:up").kind is ActionKind.SCROLL and b.lookup("window", "palm_side:down").arg("dy") == "-3"
    assert b.lookup("window", "point_up") is None  # a slide shape never fires on sight
    assert b.lookup("window", "point_up:left").arg("chord") == "ctrl+left"
    assert parse_chord(b.lookup("window", "point_up:right").arg("chord")).key == "right"
    assert b.lookup("window", "two_up") is None  # the peace sign is the media leader, nothing in the window menu
    assert b.lookup("window", "fist") is None  # fist is the escape, never bound by default


def test_config_round_trip(tmp_path: Path):
    cfg = default_config()
    cfg.timing.leader_hold_s = 1.25
    p = tmp_path / "config.toml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.timing.leader_hold_s == 1.25
    assert back.bindings().lookup("window", "h_left").name == "Previous tab"


def test_saved_config_keeps_new_default_bindings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[[namespaces.window.bindings]]\ngesture = "h_left"\naction = { kind = "key", name = "Custom", chord = "cmd+1" }\n'
    )
    b = load_config(p).bindings()
    assert b.lookup("window", "h_left").name == "Custom"  # user override wins
    assert b.lookup("window", "point_up:left").name == "Desktop left"  # a default the file did not mention is kept


def test_missing_config_gives_defaults(tmp_path: Path):
    assert load_config(tmp_path / "nope.toml").timing.leader_hold_s == 1.5  # the shipped STRICT hold
