"""The settings window's view-model: fields come from the dataclasses, edits round-trip
into a Config, bad numbers are refused, the profile popup refills the timing keys, and a
save produces a file load_config reads back."""

from dataclasses import fields
from pathlib import Path

import pytest

from visual_actions.core.config import FAST, STRICT, Config, default_config, load_config
from visual_actions.ui.settings_model import (
    SECTIONS,
    SettingsModel,
    discover_fields,
    format_value,
    parse_value,
)


def test_every_section_field_is_discovered():
    specs = discover_fields()
    by_section = {s: [f.name for f in specs if f.section == s] for s in SECTIONS}
    cfg = default_config()
    for section in SECTIONS:
        expected = [f.name for f in fields(type(getattr(cfg, section)))]
        assert by_section[section] == expected, section
    kinds = {f.key: f.kind for f in specs}
    assert kinds["timing.leader_hold_s"] == "float"
    assert kinds["timing.popup_ms"] == "int"
    assert kinds["feedback.audio"] == "bool"
    assert kinds["leader.profile"] == "choice"
    assert kinds["recognizer.model"] == "optional_str"


def test_inline_comment_becomes_help_and_choices_and_ranges_attach():
    specs = {f.key: f for f in discover_fields()}
    assert specs["timing.leader_hold_s"].help.startswith("a clear, still palm")
    assert specs["feedback.record_sessions"].help.startswith("every start writes")
    assert specs["timing.command_timeout_s"].help == ""  # no comment on that line
    assert specs["leader.profile"].choices == (STRICT, FAST)
    assert specs["drag.box_x0"].range == (0.0, 1.0)
    assert specs["drag.gain"].range == (0.2, 3.0)
    assert specs["feedback.audio"].range is None


def test_edits_round_trip_into_a_config():
    m = SettingsModel(default_config())
    assert m.set_text("timing.leader_hold_s", "2.25") is None
    assert m.set_text("feedback.audio", "false") is None
    assert m.set_text("drag.gain", "1.5") is None
    assert m.set_text("timing.popup_ms", "1200") is None
    assert m.set_text("recognizer.model", "") is None
    cfg = m.build_config()
    assert isinstance(cfg, Config)
    assert cfg.timing.leader_hold_s == 2.25
    assert cfg.feedback.audio is False
    assert cfg.drag.gain == 1.5
    assert cfg.timing.popup_ms == 1200
    assert cfg.recognizer.model is None
    assert cfg.namespaces == default_config().namespaces  # bindings pass through untouched
    assert m.dirty()
    assert set(m.diff()) == {"timing.leader_hold_s", "feedback.audio", "drag.gain", "timing.popup_ms"}
    m.revert()
    assert not m.dirty() and m.diff() == {}


def test_invalid_number_is_rejected_and_the_last_value_kept():
    m = SettingsModel(default_config())
    assert "number" in (m.set_text("timing.leader_hold_s", "fast") or "")
    assert m.value("timing.leader_hold_s") == 0.75
    assert "timing.leader_hold_s" in m.errors
    assert "whole" in (m.set_text("timing.popup_ms", "1.5") or "")
    assert m.value("timing.popup_ms") == 900
    assert m.set_text("timing.leader_hold_s", "inf") is not None
    assert m.validate()  # the errors block a save
    with pytest.raises(ValueError):
        m.save(Path("/nonexistent/config.toml"))
    assert m.set_text("timing.leader_hold_s", "1.8") is None
    assert "timing.leader_hold_s" not in m.errors


def test_cross_field_checks():
    m = SettingsModel(default_config())
    m.set_text("drag.box_x0", "0.9")
    assert any("reach box" in p for p in m.validate())
    m.set_text("drag.box_x0", "0.1")
    m.set_text("drag.pinch_on", "0.6")
    assert any("pinch" in p for p in m.validate())
    m.set_text("drag.pinch_on", "0.3")
    assert m.validate() == []


def test_profile_switch_refills_the_timing_fields():
    m = SettingsModel(default_config())
    assert m.profile == STRICT
    assert m.value("timing.leader_hold_s") == 0.75 and m.value("timing.quick_command") is False
    m.set_text("drag.gain", "2.0")  # an unrelated edit survives the switch
    changed = m.set_profile(FAST)
    assert m.profile == FAST
    assert m.value("timing.leader_hold_s") == 1.1
    assert m.value("timing.confidence_gain") == 1.5
    assert m.value("timing.quick_command") is True
    assert m.value("timing.hold_lost_grace_s") == 0.3
    assert m.value("recognizer.palm_strict") is False
    assert m.value("drag.gain") == 2.0
    assert "timing.leader_hold_s" in changed and "leader.profile" in changed
    assert set(m.diff()) == {"drag.gain"}  # against the FAST defaults only the edit differs
    m.set_profile(STRICT)
    assert m.value("timing.leader_hold_s") == 0.75 and m.value("recognizer.palm_strict") is True


def test_reset_to_profile_defaults_keeps_the_profile():
    m = SettingsModel(default_config())
    m.set_profile(FAST)
    m.set_text("timing.leader_hold_s", "2.0")
    m.set_text("feedback.preview", "false")
    m.reset_to_profile_defaults()
    assert m.profile == FAST
    assert m.value("timing.leader_hold_s") == 1.1
    assert m.value("feedback.preview") is True
    assert m.diff() == {}


def test_save_writes_a_file_load_config_reads_back(tmp_path: Path):
    path = tmp_path / "config.toml"
    m = SettingsModel(default_config())
    m.set_profile(FAST)
    m.set_text("timing.leader_hold_s", "1.3")
    m.set_text("feedback.dashboard", "false")
    m.set_text("drag.snap_enabled", "false")
    cfg = m.save(path)
    assert not m.dirty()  # the written config is the new baseline
    text = path.read_text()
    assert 'profile = "fast"' in text
    assert "leader_hold_s = 1.3" in text
    assert "command_timeout_s" not in text  # only deltas from the profile are written
    back = load_config(path)
    assert back.leader.profile == FAST
    assert back.timing.leader_hold_s == 1.3
    assert back.timing.confidence_gain == cfg.timing.confidence_gain == 1.5
    assert back.feedback.dashboard is False
    assert back.drag.snap_enabled is False
    assert back.recognizer.palm_strict is False
    # a second model loaded from the file sees the same values
    m2 = SettingsModel(back)
    assert m2.value("timing.leader_hold_s") == 1.3 and m2.profile == FAST


def test_format_and_parse_are_inverse():
    specs = {f.key: f for f in discover_fields()}
    for key, value in (("timing.leader_hold_s", 1.5), ("timing.adjust_step", 0.05), ("timing.popup_ms", 900), ("feedback.audio", True), ("recognizer.model", None), ("leader.profile", "fast")):
        spec = specs[key]
        assert parse_value(spec, format_value(spec, value)) == value
