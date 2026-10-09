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
    leader_hold_s: float = 1.5  # a clear, still palm for this long, straight
    command_timeout_s: float = 5.0
    escape_fist_s: float = 1.0
    escape_lost_s: float = 1.5
    popup_ms: int = 900
    tick_ms: int = 10  # queue drain + timers; 10 ms keeps drag moves from bunching into 50 ms bursts
    confidence_gain: float = 1.0  # hold fill rate at confidence 1.0; 1.0 = wall clock, so confidence can only slow the hold
    leader_min_confidence: float = 0.9  # palm tokens below this neither start nor fill the hold
    chain_commands: bool = True  # a menu stays open after each command (timeout restarts); fist or timeout closes it
    leader_release_tokens: int = 3  # after arming, the leader shape fires (if bound) only after this many other tokens in a row
    adjust_step: float = 0.05  # media pinch: sideways travel (fraction of frame width) per volume step
    adjust_settle_s: float = 0.25  # media pinch: hold the pinch still this long before movement changes the volume
    adjust_settle_travel: float = 0.04  # media pinch: drift during the settle that restarts it
    hold_break_tokens: int = 1  # any non-palm token abandons the hold (strict: the palm must be continuous)
    hold_reset_on_move: bool = True  # a moving palm resets the hold instead of pausing it
    quick_command: bool = False  # off: it bypasses the hold and was the main false-fire channel
    quick_command_min_hold_s: float = 0.3  # clear palm this long + a confident bound gesture = arm and fire in one motion
    quick_command_min_confidence: float = 0.85
    drag_lost_grace_s: float = 2.5  # hand lost mid-drag: window stays and waits this long for the hand
    repeat_window_s: float = 1.5  # after a repeatable action, a sideways slide of the same shape fires it again within this
    resume_grace_s: float = 0.6  # hand back mid-suspension: wait this long for the pinch before dropping
    repeat_slide: float = 0.10  # the slide: wrist must move this fraction of the frame width sideways
    # Interruptions (hand off-screen or a tracker blip). The fist is still the cancel; these only decide
    # how long an interrupted interaction waits for the hand to come back.
    keep_armed_on_lost: bool = True  # an armed or repeat window keeps its own deadline across a hand loss (see modes.py)
    lost_blip_ms: int = 150  # a loss shorter than this keeps the token smoother and pinch detector state (presence already debounces ~100 ms)
    hold_lost_grace_s: float = 0.0  # a loss shorter than this pauses the hold; 0 = reset (STRICT); FAST sets 0.3

    def to_timing(self) -> Timing:
        s = 1_000_000_000
        return Timing(
            leader_hold_ns=int(self.leader_hold_s * s),
            command_timeout_ns=int(self.command_timeout_s * s),
            escape_fist_ns=int(self.escape_fist_s * s),
            escape_lost_ns=int(self.escape_lost_s * s),
            confidence_gain=self.confidence_gain,
            leader_min_confidence=self.leader_min_confidence,
            leader_release_tokens=self.leader_release_tokens,
            chain_commands=self.chain_commands,
            adjust_step=self.adjust_step,
            adjust_settle_ns=int(self.adjust_settle_s * s),
            adjust_settle_travel=self.adjust_settle_travel,
            drag_lost_grace_ns=int(self.drag_lost_grace_s * s),
            hold_break_tokens=self.hold_break_tokens,
            hold_reset_on_move=self.hold_reset_on_move,
            quick_command=self.quick_command,
            quick_command_min_hold_ns=int(self.quick_command_min_hold_s * s),
            quick_command_min_confidence=self.quick_command_min_confidence,
            repeat_window_ns=int(self.repeat_window_s * s),
            repeat_slide=self.repeat_slide,
            resume_grace_ns=int(self.resume_grace_s * s),
            keep_armed_on_lost=self.keep_armed_on_lost,
            hold_lost_grace_ns=int(self.hold_lost_grace_s * s),
        )


@dataclass
class FeedbackConfig:
    audio: bool = True
    popup: bool = True
    preview: bool = True  # debug window: live camera with the hand skeleton (recording is unaffected)
    cursor: bool = True  # on-screen marker during pinch activity (dragging, or a brief ring on a missed pinch)
    cursor_while_armed: bool = False  # also track the hand as a ring the whole time the window is armed
    record_sessions: bool = True  # every start writes sessions/<stamp>/{video.mp4,events.log,landmarks.jsonl}
    dashboard: bool = True  # local live status page
    dashboard_port: int = 8765


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
    palm_strict: bool = True  # the model's open_palm must also pass the geometric palm rule (spread fingers, thumb out)
    fire_evidence: float = 0.9  # summed token confidence needed to fire: one sure token, or several weak ones
    min_token_confidence: float = 0.3  # tokens below this never count toward firing
    smoothing_ms: int = 250
    still_px: float = 12.0
    ambiguous_low: float = 0.45
    ambiguous_high: float = 0.75


@dataclass
class LeaderConfig:
    """Face-touch veto: a hand resting on a face reads as an open palm. Measured on the
    2026-10-08 sessions: hands on the cheek/chin sit 70-77 % inside the face box with a
    finger-tip spread of 0.34-0.52; deliberate palms beside or in front of the face stay
    under 32 % overlap or keep a spread above 0.6."""

    face_veto: bool = True
    face_overlap: float = 0.5  # hand box fraction inside a face box
    face_spread: float = 0.6  # index-tip to pinky-tip distance in hand units; below this the palm is vetoed
    profile: str = "strict"  # leader preset (strict | fast); a saved file carries this and only the keys that differ from it


@dataclass
class DragConfig:
    enabled: bool = True
    pinch_on: float = 0.3  # canonical thumb-index distance to start a pinch
    pinch_off: float = 0.55  # distance to release (hysteresis); the user's pinched p97 is 0.34, released p50 0.76
    debounce_frames: int = 2  # closed frames in a row to grab
    release_frames: int = 3  # open frames in a row to release (a carried pinch flickers open for a frame or two)
    release_grace_ms: int = 400  # a release re-grabbed within this, near the same spot, continues the drag
    regrab_px: float = 150.0  # how near (screen px, per axis) that re-grab must be
    move_deadband_px: float = 1.0  # window moves under this are skipped (no sub-pixel AX traffic)
    glitch_px: float = 24.0  # a one-frame pointer jump this big from a still hand is held a frame and dropped if it comes back
    box_x0: float = 0.15  # reach box in the camera frame mapped to the full screen
    box_x1: float = 0.85
    box_y0: float = 0.15
    box_y1: float = 0.85
    # Distance. By default (depth_gain = 0) the pointer maps the camera frame directly, so a
    # hand farther from the camera moves the pointer less per centimetre: speed follows
    # distance, which users liked. depth_gain = 1 normalizes that using ref_hand_scale, the
    # hand size (wrist to middle knuckle, as a fraction of frame width) at the distance the
    # user normally sits. `calibrate` measures ref_hand_scale from the user's recordings.
    depth_gain: float = 0.0
    ref_hand_scale: float = 0.12  # measured typical seating distance; 0.12 ~ 55 cm from a MacBook camera
    gain: float = 1.0  # window pixels per pointer pixel
    focus_on_grab: bool = True  # a grabbed window becomes active and the pane under the pinch gets keyboard focus
    smooth_min_cutoff: float = 0.6  # One Euro: lower = calmer still hand (measured 2026-10-09 on 325 drags)
    smooth_beta: float = 0.01  # One Euro speed term in screen px; 0.05 let tracker noise open the cutoff
    snap_enabled: bool = True  # our own edge snapping (BetterTouchTool and native tiling only see real mouse drags)
    snap_edge_px: float = 28.0  # pointer this close to a screen edge arms a half-screen zone
    snap_corner_px: float = 110.0  # this close to both edges arms a quarter zone
    snap_dwell_ms: int = 150  # the pointer must stay in a zone this long before it previews
    snap_quarters: bool = True
    snap_maximize: bool = True  # top edge = maximize to the visible frame


@dataclass
class PresenceConfig:
    lost_frames: int = 3  # consecutive frames without a usable hand before it is declared lost
    # Fast-exit predictor (core/presence.py): declare the hand lost at once, skipping the
    # debounce, when it is racing out of the frame or the tracker reports a phantom clamped
    # to the border. The 2026-10-08 sessions show flick-outs at 2..4.8 frame widths/s.
    fast_exit: bool = True
    fast_exit_speed: float = 2.0  # wrist or pinch-point speed, frame widths per second
    fast_exit_lookahead_frames: float = 2.0  # extrapolate this many frames; outside -> lost
    fast_exit_collapse: float = 0.5  # landmark spread under this fraction of recent spread, at a border -> phantom
    fast_exit_min_confidence: float = 0.5  # tracker confidence under this, at a border -> phantom
    fast_exit_reach: float = 0.1  # only predict when the point is already this close to that edge (frame widths)
    fast_exit_bottom: bool = False  # also predict the pinch point leaving through the bottom (hands rest there: noisy)


@dataclass
class LearnConfig:
    """The nightly learning job (learn/): sessions -> labels -> a per-user candidate model,
    promoted only when it beats the live one on held-out sessions."""

    enabled: bool = True
    hour: int = 2  # local hour the launchd job runs at
    user_weight: float = 2.0  # sample weight multiplier for this user's own frames (public / synthetic frames weigh 1.0)
    min_new_frames: int = 200  # skip training when the day exported fewer labelled frames than this
    holdout_sessions: int = 3  # newest sessions from earlier days, held out of training and replayed for the gate
    location: str = ""  # free camera / location tag written into the model manifest (e.g. "desk-macbook")
    accuracy_margin: float = 0.01  # gate: held-out weighted frame accuracy may drop at most this much
    intended_keep: float = 0.95  # gate: weak-intended fires must stay at least this fraction of the champion's
    judge_min_conf: float = 0.7  # judge tags below this confidence are not exported
    keep_versions: int = 5  # user model versions kept on disk


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
    drag: DragConfig = field(default_factory=DragConfig)
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)
    namespaces: dict[str, NamespaceConfig] = field(default_factory=dict)

    def leaders(self) -> dict[str, str]:
        """Leader gesture -> the namespace (root mode) it opens. First namespace wins a shared leader."""
        out: dict[str, str] = {}
        for ns, nc in self.namespaces.items():
            out.setdefault(nc.leader, ns)
        return out

    def bindings(self) -> Bindings:
        out = Bindings()
        for ns, nc in self.namespaces.items():
            for b in nc.bindings:
                out.add(Binding(namespace=ns, gesture=b["gesture"], action=action_from_dict(b["action"])))
        return out


DEFAULT_BINDINGS: list[dict[str, Any]] = [
    {"gesture": "h_left", "action": {"kind": "key", "name": "Previous tab", "chord": "cmd+shift+[", "repeat": True}},
    {"gesture": "h_right", "action": {"kind": "key", "name": "Next tab", "chord": "cmd+shift+]", "repeat": True}},
    # The scroll hand: right hand turned edge-on to the camera, fingers together pointing sideways,
    # arm roughly horizontal. Moving it up or down scrolls the active window, one scroll per 6 % of
    # frame height (the binding's own step), chaining. It is a slide shape: it never fires on sight.
    {"gesture": "palm_side:up", "action": {"kind": "scroll", "name": "Scroll up", "dy": 3, "step": 0.06}},
    {"gesture": "palm_side:down", "action": {"kind": "scroll", "name": "Scroll down", "dy": -3, "step": 0.06}},
    # One finger up is a slide shape: it never fires on sight; sliding it sideways by repeat_slide
    # switches desktops one step per slide (Mission Control's Ctrl+Arrow, posted with the fn flags).
    # A flick: out to one side and back to where it started is one switch; the return stroke is not
    # the opposite switch, and the next flick counts once the finger is back near the start.
    {"gesture": "point_up:left", "action": {"kind": "key", "name": "Desktop left", "chord": "ctrl+left", "flick": True}},
    {"gesture": "point_up:right", "action": {"kind": "key", "name": "Desktop right", "chord": "ctrl+right", "flick": True}},
    {"gesture": "middle_up", "action": {"kind": "open", "name": "Never gonna give you up", "target": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}},
]


# Media mode, opened by holding the peace sign (two_up). pinch_right / pinch_left are not
# hand shapes: a pinch enters the adjust state, and each adjust_step of sideways travel
# (toward the user's right or left) with the fingers pinched fires one of them.
DEFAULT_MEDIA_BINDINGS: list[dict[str, Any]] = [
    {"gesture": "point_up", "action": {"kind": "media", "name": "Play/Pause", "verb": "play_pause"}},
    # H sign pointing left = next, pointing right = previous (2026-10-08 takes: the model read
    # every bin of both correctly). Thumbs up/down stay recognized but unbound; fist stays sacred.
    {"gesture": "h_left", "action": {"kind": "media", "name": "Next track", "verb": "next"}},
    {"gesture": "h_right", "action": {"kind": "media", "name": "Previous track", "verb": "prev"}},
    {"gesture": "pinch_right", "action": {"kind": "media", "name": "Volume up", "verb": "volume_up"}},
    {"gesture": "pinch_left", "action": {"kind": "media", "name": "Volume down", "verb": "volume_down"}},
]


def _base_config() -> Config:
    return Config(
        namespaces={
            "window": NamespaceConfig(leader="open_palm", bindings=list(DEFAULT_BINDINGS)),
            "media": NamespaceConfig(leader="two_up", bindings=list(DEFAULT_MEDIA_BINDINGS)),
        }
    )


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
    # [leader] profile = "strict" | "fast" sets the leader preset first; explicit [timing] keys still override
    apply_profile(cfg, data.get("leader", {}).get("profile", STRICT))
    _merge(cfg.timing, data.get("timing", {}))
    _merge(cfg.feedback, data.get("feedback", {}))
    _merge(cfg.camera, data.get("camera", {}))
    _merge(cfg.recognizer, data.get("recognizer", {}))
    _merge(cfg.drag, data.get("drag", {}))
    _merge(cfg.leader, data.get("leader", {}))
    _merge(cfg.presence, data.get("presence", {}))
    _merge(cfg.learn, data.get("learn", {}))
    if "namespaces" in data:
        # User bindings override defaults per gesture; default bindings the file does not
        # mention are kept, so a config saved before a gesture existed still gets it.
        merged: dict[str, NamespaceConfig] = {}
        for ns, nc in data["namespaces"].items():
            user = list(nc.get("bindings", []))
            have = {b["gesture"] for b in user}
            defaults = cfg.namespaces.get(ns)
            extra = [b for b in defaults.bindings if b["gesture"] not in have] if defaults else []
            leader = nc.get("leader", defaults.leader if defaults else "open_palm")
            merged[ns] = NamespaceConfig(leader=leader, bindings=user + extra)
        for ns, nc in cfg.namespaces.items():
            merged.setdefault(ns, nc)
        cfg.namespaces = merged
    return cfg


def _drop_none(obj: Any) -> Any:
    """TOML has no null; absent keys fall back to defaults on load."""
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def _diff(cur: Any, ref: Any) -> Any:
    """The part of `cur` that differs from `ref`; None when nothing does."""
    if isinstance(cur, dict) and isinstance(ref, dict):
        out: dict[str, Any] = {}
        for k, v in cur.items():
            if k not in ref:
                if v is not None:
                    out[k] = _drop_none(v)
            else:
                d = _diff(v, ref[k])
                if d is not None:
                    out[k] = d
        return out or None
    return None if cur == ref else _drop_none(cur)


def save_config(cfg: Config, path: Path) -> None:
    """Write the profile name and only the keys that differ from that profile's defaults.
    A file that spelled out every value froze the defaults of the day it was written: the
    2026-10-08 calibration pinned the fast timings, so the strict profile never ran live."""
    import tomli_w

    profile = cfg.leader.profile
    data = _diff(asdict(cfg), asdict(default_config(profile))) or {}
    data.setdefault("leader", {})["profile"] = profile
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(data, f)


STRICT, FAST = "strict", "fast"


def apply_profile(cfg: Config, profile: str) -> Config:
    """Leader profiles. STRICT (the app default, 2026-10-08): a clear, still palm for
    1.5 s straight; confidence can only slow it; any flicker or movement resets; no
    quick command; the model's palm must also pass the geometric palm rule.
    FAST: the earlier, looser behaviour (1.1 s, confidence speeds up, pauses, quick command)."""
    t, r = cfg.timing, cfg.recognizer
    cfg.leader.profile = profile
    if profile == STRICT:
        t.leader_hold_s, t.confidence_gain, t.leader_min_confidence = 1.5, 1.0, 0.9
        t.hold_break_tokens, t.hold_reset_on_move, t.quick_command = 1, True, False
        t.hold_lost_grace_s = 0.0  # the palm must be continuous: a lost hand resets the hold
        r.palm_strict = True
    elif profile == FAST:
        t.leader_hold_s, t.confidence_gain, t.leader_min_confidence = 1.1, 1.5, 0.8
        t.hold_break_tokens, t.hold_reset_on_move, t.quick_command = 2, False, True
        t.hold_lost_grace_s = 0.3  # a brief loss pauses the hold
        r.palm_strict = False
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return cfg


def default_config(profile: str = STRICT) -> Config:
    return apply_profile(_base_config(), profile)
