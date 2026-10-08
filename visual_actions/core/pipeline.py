"""Glue from HandSeen/HandLost/Tick events to tokens, modes and actions. Runs on one thread."""

from __future__ import annotations

from .automation import DesktopAutomation
from .bindings import Bindings
from .config import Config
from .dispatcher import Dispatcher
from .drag import DragController, WindowMover
from .events import Bus, HandLost, HandSeen, Tick, TokenEmitted
from .modes import ModeEngine
from .normalize import to_user_frame
from .pinch import PinchDetector
from .pointer import PointerMap, ReachBox, SmoothedPointer
from .recognizer import CompositeRecognizer, Recognizer, RuleRecognizer, SklearnRecognizer, Smoother
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
        size = getattr(mover, "screen_size", None)
        sw, sh = size() if callable(size) else (1440, 900)
        pointer = SmoothedPointer(
            PointerMap(sw, sh, ReachBox(d.box_x0, d.box_x1, d.box_y0, d.box_y1), d.depth_gain, d.ref_hand_scale),
            d.smooth_min_cutoff,
            d.smooth_beta,
        )
        self.drag = DragController(bus, mover, pointer, gain=d.gain) if d.enabled else None
        self.engine = ModeEngine(
            bus=bus,
            bindings=bindings if bindings is not None else config.bindings(),
            timing=config.timing.to_timing(),
            fire=self._fire,
            fire_evidence=config.recognizer.fire_evidence,
            min_token_confidence=config.recognizer.min_token_confidence,
            drag=self.drag,
        )
        bus.subscribe(HandSeen, self._on_hand_seen)
        bus.subscribe(HandLost, self._on_hand_lost)
        bus.subscribe(Tick, self._on_tick)

    def _fire(self, action: Action, t_ns: int) -> None:
        self.dispatcher.dispatch(action, t_ns)

    def _on_hand_seen(self, ev: HandSeen) -> None:
        hf = to_user_frame(ev.hand_frame, self.config.camera.mirror)
        if self.drag is not None:
            pev = self.pinch.update(hf)
            if pev is not None:
                self.engine.on_pinch(pev)
        name, conf = self.recognizer.classify(hf)
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


class AutomationMover:
    """WindowMover over the DesktopAutomation window API (real or mock)."""

    def __init__(self, automation: DesktopAutomation) -> None:
        self.automation = automation

    def screen_size(self) -> tuple[int, int]:
        return self.automation.screen_size()

    def grab(self, x: float, y: float):
        return self.automation.window_at(x, y)

    def origin(self, handle) -> tuple[float, float]:
        return (float(handle.frame.x), float(handle.frame.y))

    def move(self, handle, x: float, y: float) -> bool:
        return self.automation.move_window(handle, x, y)

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
