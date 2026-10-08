"""Terminal live loop: capture thread -> queue -> main-thread pipeline, text output."""

from __future__ import annotations

import queue
import time

from .core.config import Config
from .core.dispatcher import Dispatcher
from .core.events import ActionFired, Bus, ModeChanged, Tick, TokenEmitted
from .core.pipeline import Pipeline
from .platform import factory
from .ui.capture import CaptureThread


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
    session = None
    if cfg.feedback.record_sessions:
        from .paths import sessions_dir
        from .session import SessionRecorder

        session = SessionRecorder(sessions_dir(), bus)
        print(f"recording session to {session.dir}")
    capture = CaptureThread(services.camera, q, use_gate=use_gate, sink=session)
    if cfg.feedback.dashboard:
        from .ui.dashboard import Dashboard

        try:
            dash = Dashboard(bus, cfg, stats=lambda: {"frames": capture.frames, "tracked": capture.tracked, "error": capture.error}, port=cfg.feedback.dashboard_port)
            print(f"dashboard: {dash.url}")
        except OSError as exc:
            print(f"dashboard disabled: {exc}")
    capture.start()
    tick_s = cfg.timing.tick_ms / 1000
    print("running; ctrl-c to stop")
    try:
        while capture.is_alive():
            deadline = time.monotonic() + tick_s
            while True:
                try:
                    ev = q.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                bus.publish(ev)
            bus.publish(Tick(time.monotonic_ns()))
        if capture.error:
            print("capture error:", capture.error)
            return 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        if session is not None:
            print("saved session:", session.close())
        if capture.frames:
            print(f"frames={capture.frames} tracked={capture.tracked} ({100 * capture.tracked / capture.frames:.0f}% past the gate)")
    return 0
