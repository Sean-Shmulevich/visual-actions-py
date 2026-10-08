"""Glue from HandSeen/HandLost/Tick events to tokens, modes and actions. Runs on one thread."""

from __future__ import annotations

from .bindings import Bindings
from .config import Config
from .dispatcher import Dispatcher
from .events import Bus, HandLost, HandSeen, Tick, TokenEmitted
from .modes import ModeEngine
from .normalize import to_user_frame
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
        self.engine = ModeEngine(
            bus=bus,
            bindings=bindings if bindings is not None else config.bindings(),
            timing=config.timing.to_timing(),
            fire=self._fire,
        )
        bus.subscribe(HandSeen, self._on_hand_seen)
        bus.subscribe(HandLost, self._on_hand_lost)
        bus.subscribe(Tick, self._on_tick)

    def _fire(self, action: Action, t_ns: int) -> None:
        self.dispatcher.dispatch(action, t_ns)

    def _on_hand_seen(self, ev: HandSeen) -> None:
        hf = to_user_frame(ev.hand_frame, self.config.camera.mirror)
        name, conf = self.recognizer.classify(hf)
        tok = self.smoother.push(hf, name, conf)
        if tok is not None:
            self.bus.publish(TokenEmitted(tok))
            self.engine.on_token(tok)

    def _on_hand_lost(self, ev: HandLost) -> None:
        self.smoother.reset()
        self.engine.on_hand_lost(ev.t_ns)

    def _on_tick(self, ev: Tick) -> None:
        self.engine.on_tick(ev.t_ns)


def build_recognizer(config: Config) -> Recognizer:
    rules = RuleRecognizer()
    model = None
    if config.recognizer.model:
        from pathlib import Path

        p = Path(config.recognizer.model)
        if p.exists():
            model = SklearnRecognizer(p)
    return CompositeRecognizer(rules, model)
