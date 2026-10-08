"""Live loop: capture thread -> queue -> main thread pipeline. Terminal output only in v0.1."""

from __future__ import annotations

import queue
import threading
import time

from .core.config import Config
from .core.dispatcher import Dispatcher
from .core.events import ActionFired, Bus, HandLost, HandSeen, ModeChanged, Tick, TokenEmitted
from .core.gate import AlwaysOpenGate, HandGate, MotionSkinGate
from .core.pipeline import Pipeline
from .core.tracker import MediaPipeTracker
from .paths import model_path
from .platform import factory


def run_live(cfg: Config, dry_run: bool, use_gate: bool = True, verbose: bool = True) -> int:
    services = factory.create(cfg.camera, dry_run=dry_run)
    status = services.permissions.check(prompt=True)
    print(f"platform={services.name} camera={status.camera.value} accessibility={status.accessibility.value}")

    bus = Bus()
    Pipeline(bus, cfg, Dispatcher(bus, services.automation))
    if cfg.feedback.audio:
        from .ui.sound import SoundFeedback

        SoundFeedback(bus)
    if verbose:
        bus.subscribe(TokenEmitted, lambda e: print(f"token {e.token.name:10s} conf={e.token.confidence:.2f} still={e.token.still}"))
        bus.subscribe(ModeChanged, lambda e: print(f"mode  {e.old} -> {e.new}"))
        bus.subscribe(ActionFired, lambda e: print(f"ACTION {e.action.name} ok={e.ok} {e.message}"))

    q: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()
    gate: HandGate = MotionSkinGate() if use_gate else AlwaysOpenGate()

    def capture() -> None:
        tracker = MediaPipeTracker(model_path())
        services.camera.open()
        seen = False
        try:
            while not stop.is_set():
                r = services.camera.read()
                if r is None:
                    continue
                t_ns, frame = r
                if not gate.open(frame, t_ns):
                    if seen:
                        q.put(HandLost(t_ns))
                        seen = False
                    continue
                hands = tracker.track(frame, t_ns)
                if hands:
                    q.put(HandSeen(hands[0]))
                    seen = True
                elif seen:
                    q.put(HandLost(t_ns))
                    seen = False
        finally:
            services.camera.close()
            tracker.close()

    th = threading.Thread(target=capture, name="capture", daemon=True)
    th.start()
    tick_s = cfg.timing.tick_ms / 1000
    print("running; ctrl-c to stop")
    try:
        while th.is_alive():
            deadline = time.monotonic() + tick_s
            while True:
                try:
                    ev = q.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                bus.publish(ev)
            bus.publish(Tick(time.monotonic_ns()))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        th.join(timeout=2)
    return 0
