"""Config dataclasses, defaults, and TOML load/save."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bindings import Bindings
from .modes import Timing
from .types import Action, ActionKind, Binding


@dataclass
class TimingConfig:
    leader_hold_s: float = 2.0
    command_timeout_s: float = 5.0
    escape_fist_s: float = 1.0
    escape_lost_s: float = 1.5
    popup_ms: int = 900
    tick_ms: int = 50

    def to_timing(self) -> Timing:
        s = 1_000_000_000
        return Timing(
            leader_hold_ns=int(self.leader_hold_s * s),
            command_timeout_ns=int(self.command_timeout_s * s),
            escape_fist_ns=int(self.escape_fist_s * s),
            escape_lost_ns=int(self.escape_lost_s * s),
        )


@dataclass
class FeedbackConfig:
    audio: bool = True
    popup: bool = True


@dataclass
class CameraConfig:
    index: int = 0
    mirror: bool = True
    width: int = 640
    height: int = 480


@dataclass
class RecognizerConfig:
    model: str | None = None  # path to a joblib; None = rules only
    rule_min: float = 0.9  # a rule needs this confidence to override the model
    smoothing_ms: int = 250
    still_px: float = 12.0
    ambiguous_low: float = 0.45
    ambiguous_high: float = 0.75


@dataclass
class NamespaceConfig:
    leader: str = "open_palm"
    bindings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Config:
    timing: TimingConfig = field(default_factory=TimingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    namespaces: dict[str, NamespaceConfig] = field(default_factory=dict)

    def bindings(self) -> Bindings:
        out = Bindings()
        for ns, nc in self.namespaces.items():
            for b in nc.bindings:
                out.add(Binding(namespace=ns, gesture=b["gesture"], action=action_from_dict(b["action"])))
        return out


DEFAULT_BINDINGS: list[dict[str, Any]] = [
    {"gesture": "h_left", "action": {"kind": "key", "name": "Cmd+Tab", "chord": "cmd+tab"}},
    {"gesture": "h_right", "action": {"kind": "key", "name": "Cmd+Shift+Tab", "chord": "cmd+shift+tab"}},
]


def default_config() -> Config:
    return Config(namespaces={"window": NamespaceConfig(bindings=list(DEFAULT_BINDINGS))})


def action_from_dict(d: dict[str, Any]) -> Action:
    kind = ActionKind(d["kind"])
    name = str(d.get("name", kind.value))
    args = tuple((k, str(v)) for k, v in d.items() if k not in ("kind", "name"))
    return Action(kind=kind, name=name, args=args)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    for k, v in data.items():
        if hasattr(dc, k):
            setattr(dc, k, v)
    return dc


def load_config(path: Path | None) -> Config:
    cfg = default_config()
    if path is None or not path.exists():
        return cfg
    with path.open("rb") as f:
        data = tomllib.load(f)
    _merge(cfg.timing, data.get("timing", {}))
    _merge(cfg.feedback, data.get("feedback", {}))
    _merge(cfg.camera, data.get("camera", {}))
    _merge(cfg.recognizer, data.get("recognizer", {}))
    if "namespaces" in data:
        cfg.namespaces = {
            ns: NamespaceConfig(leader=nc.get("leader", "open_palm"), bindings=list(nc.get("bindings", [])))
            for ns, nc in data["namespaces"].items()
        }
    return cfg


def _drop_none(obj: Any) -> Any:
    """TOML has no null; absent keys fall back to defaults on load."""
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def save_config(cfg: Config, path: Path) -> None:
    import tomli_w

    data = _drop_none(asdict(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(data, f)
