"""View-model for the settings window: the form is derived from the config dataclasses.

Pure Python, no AppKit, so the discovery, parsing, validation and profile logic is
testable on every platform. `ui/settings.py` renders it with PyObjC.

Each editable section of `Config` is a dataclass; every field becomes a `FieldSpec`
(name, kind, help text from the inline `# comment`, optional choices or slider range),
so a new config key shows up in the window without a UI edit.
"""

from __future__ import annotations

import copy
import inspect
import math
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..core.config import FAST, STRICT, Config, apply_profile, default_config, save_config

SECTIONS: tuple[str, ...] = ("leader", "timing", "feedback", "drag", "presence", "recognizer", "camera")
SECTION_TITLES = {
    "leader": "Leader",
    "timing": "Timing",
    "feedback": "Feedback",
    "drag": "Drag",
    "presence": "Presence",
    "recognizer": "Recognizer",
    "camera": "Camera",
}

# str fields with a known set of values become a popup
CHOICES: dict[str, tuple[str, ...]] = {"leader.profile": (STRICT, FAST)}

# Number fields with an obvious range get a slider beside the text field.
RANGES: dict[str, tuple[float, float]] = {
    "timing.leader_hold_s": (0.3, 3.0),
    "timing.command_timeout_s": (1.0, 15.0),
    "timing.escape_fist_s": (0.2, 3.0),
    "timing.escape_lost_s": (0.2, 5.0),
    "timing.confidence_gain": (0.5, 3.0),
    "timing.drag_lost_grace_s": (0.0, 5.0),
    "timing.repeat_window_s": (0.3, 5.0),
    "drag.gain": (0.2, 3.0),
    "drag.depth_gain": (0.0, 1.0),
    "leader.face_overlap": (0.0, 1.0),
    "leader.face_spread": (0.0, 1.5),
    "presence.fast_exit_speed": (0.5, 6.0),
    "presence.fast_exit_lookahead_frames": (0.0, 5.0),
    "recognizer.smoothing_ms": (0, 1000),
}
_UNIT_RANGE_PREFIXES = ("drag.box_", "drag.pinch_", "presence.fast_exit_collapse", "presence.fast_exit_reach")

RESTART_NOTE = (
    "Save writes the config file and restarts the capture pipeline: the camera reopens, "
    "the current session recording is closed and a new one starts. Every key takes effect "
    "that way; nothing needs the app itself restarted."
)

Kind = str  # "bool" | "int" | "float" | "str" | "optional_str" | "choice"


@dataclass(frozen=True)
class FieldSpec:
    section: str
    name: str
    kind: Kind
    help: str = ""
    choices: tuple[str, ...] = ()
    range: tuple[float, float] | None = None

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")

    @property
    def numeric(self) -> bool:
        return self.kind in ("int", "float")


_COMMENT_RE = re.compile(r"^\s*(?P<name>\w+)\s*:[^#\n]*#\s*(?P<help>.+?)\s*$")


def _inline_comments(cls: type) -> dict[str, str]:
    """`name: type = default  # help` -> {name: help}; empty when the source is unavailable."""
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    out: dict[str, str] = {}
    for line in src.splitlines():
        m = _COMMENT_RE.match(line)
        if m:
            out[m.group("name")] = m.group("help")
    return out


def _kind_of(tp: Any) -> Kind | None:
    if tp is bool:
        return "bool"
    if tp is int:
        return "int"
    if tp is float:
        return "float"
    if tp is str:
        return "str"
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = set(typing.get_args(tp))
        if args == {str, type(None)}:
            return "optional_str"
    return None


def _range_of(key: str) -> tuple[float, float] | None:
    if key in RANGES:
        return RANGES[key]
    if key.startswith(_UNIT_RANGE_PREFIXES):
        return (0.0, 1.0)
    return None


def discover_fields(cfg: Config | None = None, sections: tuple[str, ...] = SECTIONS) -> list[FieldSpec]:
    """Every scalar field of every section dataclass, in declaration order."""
    cfg = cfg or default_config()
    specs: list[FieldSpec] = []
    for section in sections:
        cls = type(getattr(cfg, section))
        hints = typing.get_type_hints(cls)
        comments = _inline_comments(cls)
        for f in fields(cls):
            key = f"{section}.{f.name}"
            kind = _kind_of(hints.get(f.name))
            if kind is None:
                continue  # lists/dicts (bindings) are not form fields
            choices = CHOICES.get(key, ())
            if choices:
                kind = "choice"
            specs.append(
                FieldSpec(
                    section=section,
                    name=f.name,
                    kind=kind,
                    help=comments.get(f.name, ""),
                    choices=choices,
                    range=_range_of(key) if kind in ("int", "float") else None,
                )
            )
    return specs


def parse_value(spec: FieldSpec, text: str) -> Any:
    """Text from a control -> a typed value; ValueError with a short message when invalid."""
    s = text.strip()
    if spec.kind == "bool":
        if s.lower() in ("1", "true", "yes", "on"):
            return True
        if s.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{spec.label}: expected true or false")
    if spec.kind == "int":
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"{spec.label}: expected a whole number") from None
    if spec.kind == "float":
        try:
            v = float(s)
        except ValueError:
            raise ValueError(f"{spec.label}: expected a number") from None
        if not math.isfinite(v):
            raise ValueError(f"{spec.label}: expected a finite number")
        return v
    if spec.kind == "choice":
        if s not in spec.choices:
            raise ValueError(f"{spec.label}: expected one of {', '.join(spec.choices)}")
        return s
    if spec.kind == "optional_str":
        return s or None
    return s


def format_value(spec: FieldSpec, value: Any) -> str:
    """A typed value -> the text a control shows."""
    if value is None:
        return ""
    if spec.kind == "bool":
        return "true" if value else "false"
    if spec.kind == "float":
        return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:.1f}"
    return str(value)


class SettingsModel:
    """Edits a copy of the config as a flat {key: value} map, validates, diffs against the
    profile defaults and saves. `errors` holds the keys whose text could not be parsed;
    their last valid value is kept so a stray keystroke never corrupts the config."""

    def __init__(self, cfg: Config) -> None:
        self.specs = discover_fields(cfg)
        self._by_key = {s.key: s for s in self.specs}
        self.saved = copy.deepcopy(cfg)
        self.values: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self._load(self.saved)

    # -- discovery ---------------------------------------------------------------

    def sections(self) -> list[str]:
        return [s for s in SECTIONS if any(f.section == s for f in self.specs)]

    def fields_in(self, section: str) -> list[FieldSpec]:
        return [f for f in self.specs if f.section == section]

    def spec(self, key: str) -> FieldSpec:
        return self._by_key[key]

    # -- values ------------------------------------------------------------------

    def _load(self, cfg: Config) -> None:
        self.values = {f.key: getattr(getattr(cfg, f.section), f.name) for f in self.specs}
        self.errors = {}

    @property
    def profile(self) -> str:
        return str(self.values["leader.profile"])

    def value(self, key: str) -> Any:
        return self.values[key]

    def text(self, key: str) -> str:
        return format_value(self._by_key[key], self.values[key])

    def set_value(self, key: str, value: Any) -> None:
        self.values[key] = value
        self.errors.pop(key, None)

    def set_text(self, key: str, text: str) -> str | None:
        """Parse `text` into the field; returns the error message, or None when accepted."""
        spec = self._by_key[key]
        try:
            self.set_value(key, parse_value(spec, text))
        except ValueError as exc:
            self.errors[key] = str(exc)
            return str(exc)
        return None

    def set_profile(self, profile: str) -> list[str]:
        """Switch the leader profile and refill the keys that profile owns (timing and
        palm_strict) via apply_profile; returns the keys whose value changed."""
        before = dict(self.values)
        cfg = apply_profile(self.build_config(), profile)
        self._load(cfg)
        return [k for k in self.values if self.values[k] != before.get(k)]

    def reset_to_profile_defaults(self) -> None:
        """Every section back to the current profile's defaults (bindings are untouched)."""
        self._load(default_config(self.profile))

    def revert(self) -> None:
        """Discard edits: back to the config that was loaded or last saved."""
        self._load(self.saved)

    # -- config ------------------------------------------------------------------

    def build_config(self) -> Config:
        """The edited values written into a copy of the saved config; invalid fields keep
        their last valid value."""
        cfg = copy.deepcopy(self.saved)
        for f in self.specs:
            setattr(getattr(cfg, f.section), f.name, self.values[f.key])
        return cfg

    def diff(self) -> dict[str, tuple[Any, Any]]:
        """{key: (edited value, profile default)} for the keys that differ from the profile."""
        ref = default_config(self.profile)
        out: dict[str, tuple[Any, Any]] = {}
        for f in self.specs:
            d = getattr(getattr(ref, f.section), f.name)
            if self.values[f.key] != d:
                out[f.key] = (self.values[f.key], d)
        return out

    def dirty(self) -> bool:
        return any(self.values[f.key] != getattr(getattr(self.saved, f.section), f.name) for f in self.specs)

    def validate(self) -> list[str]:
        """Cross-field checks on top of per-field parsing; a list of messages, empty when fine."""
        msgs = list(self.errors.values())
        v = self.values
        if v["drag.box_x0"] >= v["drag.box_x1"] or v["drag.box_y0"] >= v["drag.box_y1"]:
            msgs.append("drag: the reach box must have x0 < x1 and y0 < y1")
        if v["drag.pinch_on"] >= v["drag.pinch_off"]:
            msgs.append("drag: pinch on must be below pinch off (hysteresis)")
        if v["timing.tick_ms"] <= 0:
            msgs.append("timing: tick ms must be positive")
        for key in ("timing.leader_hold_s", "timing.command_timeout_s", "camera.width", "camera.height"):
            if v[key] <= 0:
                msgs.append(f"{self._by_key[key].label}: must be positive")
        if not (1 <= v["feedback.dashboard_port"] <= 65535):
            msgs.append("feedback: dashboard port must be 1..65535")
        return msgs

    def save(self, path: Path, writer: Callable[[Config, Path], None] = save_config) -> Config:
        """Validate, write (profile + deltas only, via core.config.save_config) and make the
        written config the new baseline for `revert`. Raises ValueError when invalid."""
        problems = self.validate()
        if problems:
            raise ValueError("; ".join(problems))
        cfg = self.build_config()
        writer(cfg, path)
        self.saved = copy.deepcopy(cfg)
        return cfg
