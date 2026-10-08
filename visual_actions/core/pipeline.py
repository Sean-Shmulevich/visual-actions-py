"""Glue from HandSeen/HandLost/Tick events to tokens, modes and actions. Runs on one thread."""

from __future__ import annotations

from .automation import DesktopAutomation, Rect
from .bindings import Bindings
from .config import Config
from .dispatcher import Dispatcher
from .drag import DragController, WindowMover
from .events import Bus, HandLost, HandSeen, PalmVetoed, PointerMoved, Tick, TokenEmitted
from .modes import ARMED, DRAGGING, ModeEngine
from .normalize import to_user_frame
from .pinch import PinchDetector
from .pointer import PointerMap, ReachBox, SmoothedPointer
from .recognizer import (
    NONE,
    OPEN_PALM,
    CompositeRecognizer,
    Recognizer,
    RuleRecognizer,
    SklearnRecognizer,
    Smoother,
    tip_spread,
)
from .snap import SnapEngine, SnapRules
from .types import Action


class Pipeline:
    def __init__(
        self,
        bus: Bus,
        config: Config,
        dispatcher: Dispatcher,
        recognizer: Recognizer | None = None,
        bindings: Bindings | None = None,
        mover: WindowMover | None = None,
    ) -> None:
        self.bus = bus
        self.config = config
        self.dispatcher = dispatcher
        self.recognizer = recognizer or build_recognizer(config)
        self.smoother = Smoother(
            window_ns=config.recognizer.smoothing_ms * 1_000_000,
            still_px=config.recognizer.still_px,
            frame_width_px=config.camera.width,
        )
        d = config.drag
        self.pinch = PinchDetector(d.pinch_on, d.pinch_off, d.debounce_frames)
        if mover is None:
            mover = AutomationMover(dispatcher.automation)
        sw, sh = _screen_size(mover)
        pointer = SmoothedPointer(
            PointerMap(sw, sh, ReachBox(d.box_x0, d.box_x1, d.box_y0, d.box_y1), d.depth_gain, d.ref_hand_scale),
            d.smooth_min_cutoff,
            d.smooth_beta,
        )
        snap = None
        if d.enabled and d.snap_enabled:
            visible = _visible_frame(mover, sw, sh)
            snap = SnapEngine(
                visible,
                SnapRules(
                    edge_px=d.snap_edge_px,
                    corner_px=d.snap_corner_px,
                    dwell_ns=d.snap_dwell_ms * 1_000_000,
                    quarters=d.snap_quarters,
                    maximize=d.snap_maximize,
                ),
            )
        self.drag = (
            DragController(bus, mover, pointer, gain=d.gain, snap=snap, focus_on_grab=d.focus_on_grab) if d.enabled else None
        )
        self.engine = ModeEngine(
            bus=bus,
            bindings=bindings if bindings is not None else config.bindings(),
            timing=config.timing.to_timing(),
            fire=self._fire,
            fire_evidence=config.recognizer.fire_evidence,
            min_token_confidence=config.recognizer.min_token_confidence,
            drag=self.drag,
            leaders=config.leaders(),
        )
        self._last_veto_ns = -(10**18)
        bus.subscribe(HandSeen, self._on_hand_seen)
        bus.subscribe(HandLost, self._on_hand_lost)
        bus.subscribe(Tick, self._on_tick)

    def _fire(self, action: Action, t_ns: int) -> None:
        self.dispatcher.dispatch(action, t_ns)

    def _on_hand_seen(self, ev: HandSeen) -> None:
        hf = to_user_frame(ev.hand_frame, self.config.camera.mirror)
        pev = self.pinch.update(hf)
        if pev is not None:
            self.engine.on_pinch(pev)
        if self.drag is not None and (
            self.engine.state == DRAGGING or (self.engine.state == ARMED and self.engine.drag_allowed)
        ):
            from .pinch import hand_scale, pinch_point

            px, py = pinch_point(hf)
            cx, cy = self.drag.pointer.pmap.to_screen(px, py, hand_scale(hf))
            self.bus.publish(PointerMoved(hf.t_ns, cx, cy, self.engine.state == DRAGGING))
        name, conf = self.recognizer.classify(hf)
        if name == OPEN_PALM and palm_vetoed(hf, ev.face_overlap, self.config):
            name, conf = NONE, 0.5  # a hand on a face is not a leader
            if hf.t_ns - self._last_veto_ns >= 500_000_000:
                self._last_veto_ns = hf.t_ns
                self.bus.publish(PalmVetoed(hf.t_ns, ev.face_overlap, tip_spread(hf)))
        tok = self.smoother.push(hf, name, conf)
        if tok is not None:
            self.bus.publish(TokenEmitted(tok))
            self.engine.on_token(tok)

    def _on_hand_lost(self, ev: HandLost) -> None:
        self.smoother.reset()
        self.pinch.reset()
        self.engine.on_hand_lost(ev.t_ns)

    def _on_tick(self, ev: Tick) -> None:
        self.engine.on_tick(ev.t_ns)


def palm_vetoed(hf, face_overlap: float, config: Config) -> bool:
    """Face-touch veto: the hand box is mostly inside a face box AND the fingers are not spread."""
    lc = config.leader
    return lc.face_veto and face_overlap >= lc.face_overlap and tip_spread(hf) < lc.face_spread


def _visible_frame(mover: object, sw: int, sh: int) -> Rect:
    vf = getattr(mover, "visible_frame", None)
    if callable(vf):
        r = vf()
        if isinstance(r, Rect):
            return r
    return Rect(0, 0, sw, sh)


def _screen_size(mover: object) -> tuple[int, int]:
    size = getattr(mover, "screen_size", None)
    if callable(size):
        result: tuple[int, int] = tuple(int(v) for v in size())  # type: ignore[assignment]
        return result
    return (1440, 900)


class AutomationMover:
    """WindowMover over the DesktopAutomation window API (real or mock)."""

    def __init__(self, automation: DesktopAutomation) -> None:
        self.automation = automation

    def screen_size(self) -> tuple[int, int]:
        return self.automation.screen_size()

    def visible_frame(self) -> Rect:
        return self.automation.visible_frame()

    def grab(self, x: float, y: float):
        win = self.automation.window_at(x, y)
        if win is not None:
            # Warm the driver's per-window cache now (the first AX resolve costs ~80 ms on
            # macOS) so the first real move does not hitch. A move to the current origin.
            self.automation.move_window(win, float(win.frame.x), float(win.frame.y))
        return win

    def frame(self, handle) -> Rect:
        return handle.frame

    def move(self, handle, x: float, y: float) -> bool:
        return self.automation.move_window(handle, x, y)

    def focus(self, handle, x: float, y: float) -> bool:
        return self.automation.focus_at(x, y)

    def set_frame(self, handle, rect: Rect) -> bool:
        ok_size = self.automation.resize_window(handle, float(rect.w), float(rect.h))
        ok_pos = self.automation.move_window(handle, float(rect.x), float(rect.y))
        # some apps clamp size until positioned; set size again after the move
        ok_size = self.automation.resize_window(handle, float(rect.w), float(rect.h)) or ok_size
        return ok_pos and ok_size

    def label(self, handle) -> str:
        return f"{handle.app}: {handle.title}" if handle.title else handle.app


def build_recognizer(config: Config) -> Recognizer:
    rules = RuleRecognizer()
    model = None
    if config.recognizer.model:
        from pathlib import Path

        p = Path(config.recognizer.model)
        if p.exists():
            model = SklearnRecognizer(p)
    return CompositeRecognizer(rules, model, rule_min=config.recognizer.rule_min)
