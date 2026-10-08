"""rumps menu bar shell. Owns the main thread; a rumps.Timer drives the tick loop."""

from __future__ import annotations

import queue
import subprocess
import time

import rumps

from ..core.config import Config
from ..core.dispatcher import Dispatcher
from ..core.events import ActionFired, Bus, ModeChanged, Tick
from ..core.pipeline import Pipeline
from ..paths import config_path
from ..platform import factory
from .capture import CaptureThread


class VisualActionsApp(rumps.App):
    def __init__(self, cfg: Config, dry_run: bool, use_gate: bool) -> None:
        super().__init__("Visual Actions", title="✋", quit_button=None)
        self.cfg = cfg
        self.dry_run = dry_run
        self.use_gate = use_gate
        self.services = factory.create(cfg.camera, dry_run=dry_run)
        self.bus = Bus()
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.capture: CaptureThread | None = None

        Pipeline(self.bus, cfg, Dispatcher(self.bus, self.services.automation))
        if cfg.feedback.audio:
            from .sound import SoundFeedback

            SoundFeedback(self.bus)
        if cfg.feedback.popup:
            from .overlay import Overlay

            t = cfg.timing.to_timing()
            Overlay(self.bus, t.leader_hold_ns, t.command_timeout_ns, cfg.timing.popup_ms)

        self.status_item = rumps.MenuItem("Status: starting")
        self.perm_item = rumps.MenuItem("Permissions…", callback=self.show_permissions)
        self.toggle_item = rumps.MenuItem("Stop", callback=self.toggle)
        self.dry_item = rumps.MenuItem("Dry run (print only)", callback=self.toggle_dry)
        self.dry_item.state = dry_run
        self.menu = [
            self.status_item,
            self.perm_item,
            None,
            self.toggle_item,
            self.dry_item,
            rumps.MenuItem("Open config file", callback=lambda _: subprocess.run(["open", "-t", str(config_path())], check=False)),
            None,
            rumps.MenuItem("Quit", callback=self.quit),
        ]
        self.bus.subscribe(ModeChanged, lambda e: self._set_title({"idle": "✋", "holding": "⏳", "armed": "🟢", "dragging": "🤏"}.get(e.new, "✋")))
        self.bus.subscribe(ActionFired, lambda e: self._set_status(f"Last: {e.action.name} {'ok' if e.ok else e.message}"))
        self.timer = rumps.Timer(self.tick, cfg.timing.tick_ms / 1000)

    # -- lifecycle ----------------------------------------------------------

    def run(self, **kw):  # type: ignore[override]
        self.start()
        self.timer.start()
        super().run(**kw)

    def start(self) -> None:
        status = self.services.permissions.check(prompt=True)
        self._set_status(f"camera {status.camera.value}, accessibility {status.accessibility.value}")
        self.capture = CaptureThread(self.services.camera, self.q, use_gate=self.use_gate)
        self.capture.start()
        self.toggle_item.title = "Stop"

    def stop(self) -> None:
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.toggle_item.title = "Start"
        self._set_title("✋")

    def tick(self, _timer) -> None:
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            self.bus.publish(ev)
        self.bus.publish(Tick(time.monotonic_ns()))
        if self.capture and self.capture.error:
            self._set_status(f"camera error: {self.capture.error}")
            self.stop()

    # -- menu callbacks -------------------------------------------------------

    def toggle(self, _item) -> None:
        if self.capture:
            self.stop()
        else:
            self.start()

    def toggle_dry(self, item) -> None:
        self.dry_run = not self.dry_run
        item.state = self.dry_run
        self.services = factory.create(self.cfg.camera, dry_run=self.dry_run)
        # re-point the dispatcher at the new driver without rebuilding the pipeline
        for handler_list in self.bus._handlers.values():
            for h in handler_list:
                owner = getattr(h, "__self__", None)
                if isinstance(owner, Pipeline):
                    owner.dispatcher.automation = self.services.automation
        self._set_status("dry run " + ("on" if self.dry_run else "off"))

    def show_permissions(self, _item) -> None:
        s = self.services.permissions.check(prompt=True)
        if rumps.alert(
            "Permissions",
            f"Camera: {s.camera.value}\nAccessibility: {s.accessibility.value}\n\n{s.settings_hint}",
            ok="Open Camera settings",
            cancel="Close",
        ):
            self.services.permissions.open_settings("camera")

    def quit(self, _item) -> None:
        self.stop()
        rumps.quit_application()

    # -- helpers --------------------------------------------------------------

    def _set_title(self, t: str) -> None:
        self.title = t

    def _set_status(self, s: str) -> None:
        self.status_item.title = f"Status: {s}"


def run_menubar(cfg: Config, dry_run: bool, use_gate: bool) -> int:
    VisualActionsApp(cfg, dry_run, use_gate).run()
    return 0
