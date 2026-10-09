"""rumps menu bar shell. Owns the main thread; a rumps.Timer drives the tick loop."""

from __future__ import annotations

import queue
import subprocess
import time
from typing import TYPE_CHECKING

import rumps

from ..core.config import Config
from ..core.dispatcher import Dispatcher
from ..core.events import ActionFired, Bus, ModeChanged, Tick
from ..core.pipeline import Pipeline
from ..core.presence import PresenceFilter
from ..paths import config_path
from ..platform import factory
from ..session import SessionRecorder
from .capture import CaptureThread
from .preview import DebugPreview

if TYPE_CHECKING:
    from .dashboard import Dashboard
    from .settings import SettingsWindow


class VisualActionsApp(rumps.App):
    def __init__(self, cfg: Config, dry_run: bool, use_gate: bool) -> None:
        super().__init__("Visual Actions", title="✋", quit_button=None)
        self.cfg = cfg
        self.dry_run = dry_run
        self.use_gate = use_gate
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.capture: CaptureThread | None = None
        self.preview: DebugPreview | None = None
        self.session: SessionRecorder | None = None
        self.settings: SettingsWindow | None = None
        self.status_item = rumps.MenuItem("Status: starting")
        self._feedback: list[object] = []  # SoundFeedback / CursorOverlay / Overlay, closed on a rebuild
        self.dashboard: Dashboard | None = None
        self._build(cfg)

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
            rumps.MenuItem("Settings…", callback=self.show_settings),
            rumps.MenuItem("Open config file", callback=lambda _: subprocess.run(["open", "-t", str(config_path())], check=False)),
            rumps.MenuItem("Open dashboard", callback=self.open_dashboard),
            rumps.MenuItem("Open sessions folder", callback=self.open_sessions),
            None,
            rumps.MenuItem("Quit", callback=self.quit),
        ]
        self.timer = rumps.Timer(self.tick, cfg.timing.tick_ms / 1000)
        self._install_terminate_hook()

    def _build(self, cfg: Config) -> None:
        """Everything that reads the config when it is constructed: the platform driver, a fresh
        bus, the pipeline, the feedback listeners and the dashboard. `apply_config` tears the
        previous set down and calls this again, so a saved config is applied by one rebuild
        instead of per-key live updates."""
        self.cfg = cfg
        self.services = factory.create(cfg.camera, dry_run=self.dry_run)
        self.bus = Bus()
        self.pipeline = Pipeline(self.bus, cfg, Dispatcher(self.bus, self.services.automation))
        self._feedback = []
        if cfg.feedback.audio:
            from .sound import SoundFeedback

            self._feedback.append(SoundFeedback(self.bus))
        if cfg.feedback.cursor:
            from .cursor import CursorOverlay

            self._feedback.append(CursorOverlay(self.bus, while_armed=cfg.feedback.cursor_while_armed))
        if cfg.feedback.popup:
            from .overlay import Overlay

            t = cfg.timing.to_timing()
            self._feedback.append(
                Overlay(
                    self.bus,
                    t.leader_hold_ns,
                    t.command_timeout_ns,
                    cfg.timing.popup_ms,
                    t.drag_lost_grace_ns,
                    volume=self._system_volume,
                    timing=t,
                )
            )

        self.dashboard = None
        if cfg.feedback.dashboard:
            from .dashboard import Dashboard

            try:
                self.dashboard = Dashboard(self.bus, cfg, stats=self._capture_stats, port=cfg.feedback.dashboard_port)
            except OSError as exc:  # port busy: run without it
                print(f"dashboard disabled: {exc}")
        self.bus.subscribe(ModeChanged, lambda e: self._set_title(mode_icon(e.new, e.namespace)))
        self.bus.subscribe(ActionFired, lambda e: self._set_status(f"Last: {e.action.name} {'ok' if e.ok else e.message}"))

    def _teardown(self) -> None:
        """Hide the panels and free the dashboard port; the old bus and pipeline go with the references."""
        for listener in self._feedback:
            close = getattr(listener, "close", None)
            if callable(close):
                close()
        self._feedback = []
        if self.dashboard is not None:
            self.dashboard.close()
            self.dashboard = None
        while True:  # events the old capture thread left behind belong to the old pipeline
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def apply_config(self, cfg: Config) -> None:
        """The settings window saved: stop the capture, rebuild from the new config, start again
        if it was running. Every key is applied this way (the camera reopens and a new session
        recording starts); none is patched live."""
        running = self.capture is not None
        self.stop()
        self._teardown()
        self._build(cfg)
        if self.timer.is_alive():
            self.timer.stop()
            self.timer.interval = cfg.timing.tick_ms / 1000
            self.timer.start()
        else:
            self.timer.interval = cfg.timing.tick_ms / 1000
        if running:
            self.start()

    def _system_volume(self) -> tuple[float, bool] | None:
        # looked up on each call: a reload swaps self.services
        read = getattr(self.services.automation, "volume", None)
        result = read() if callable(read) else None
        return result  # type: ignore[return-value]

    # -- lifecycle ----------------------------------------------------------

    def run(self, **kw):  # type: ignore[override]
        self.start()
        self.timer.start()
        super().run(**kw)

    def start(self) -> None:
        status = self.services.permissions.check(prompt=True)
        self._set_status(f"camera {status.camera.value}, accessibility {status.accessibility.value}")
        self.session = None
        if self.cfg.feedback.record_sessions:
            from ..paths import sessions_dir

            self.session = SessionRecorder(sessions_dir(), self.bus, config=self.cfg, meta=self._session_meta())
        self.preview = DebugPreview() if self.cfg.feedback.preview else None
        self.capture = CaptureThread(self.services.camera, self.q, use_gate=self.use_gate, sink=self.session, face_veto=self.cfg.leader.face_veto, preview=self.preview, presence=PresenceFilter.from_config(self.cfg.presence))
        self.capture.start()
        self.toggle_item.title = "Stop"

    def stop(self) -> None:
        if self.capture:
            self.capture.stop()
            self.capture = None
        if self.preview is not None:
            self.preview.close()
            self.preview = None
        if self.session is not None:
            summary = self.session.close()
            self.session = None
            self._set_status(f"saved session {summary['seconds']}s, {summary['frames']} frames")
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
        if self.preview is not None:
            self.preview.label = self.title or ""
            self.preview.show()
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

    def show_settings(self, _item) -> None:
        """One window for the life of the app: closing hides it, this brings it back."""
        if self.settings is None:
            from .settings import SettingsWindow
            from .settings_model import SettingsModel

            self.settings = SettingsWindow(SettingsModel(self.cfg), on_save=self.apply_config, path=config_path())
        self.settings.show()

    def open_sessions(self, _item) -> None:
        from ..paths import sessions_dir

        sessions_dir().mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(sessions_dir())], check=False)

    def open_dashboard(self, _item) -> None:
        if self.dashboard is None:
            rumps.alert("Dashboard", "The dashboard is disabled in config or its port was busy.")
            return
        subprocess.run(["open", self.dashboard.url], check=False)

    def _session_meta(self) -> dict:
        """What this process knows for the session's meta.json: the screen and the model in use."""
        meta: dict = {"dry_run": self.dry_run, "model_path": None, "screen": None}
        path = getattr(self.pipeline.recognizer, "model_path", None)
        meta["model_path"] = str(path) if path is not None else None
        try:
            meta["screen"] = list(self.services.automation.screen_size())
        except Exception:  # noqa: BLE001 - best-effort
            pass
        return meta

    def _capture_stats(self) -> dict:
        c = self.capture
        if c is None:
            return {"frames": 0, "tracked": 0, "error": "capture not running"}
        return {"frames": c.frames, "tracked": c.tracked, "error": c.error}

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

    def _install_terminate_hook(self) -> None:
        """Cmd-Q, Dock quit, log out and shutdown all go through NSApplicationWillTerminate;
        rumps does not forward it, so observe the notification ourselves and close the session."""
        import objc
        from AppKit import NSApplicationWillTerminateNotification
        from Foundation import NSNotificationCenter, NSObject

        app = self

        class _Observer(NSObject):
            def onTerminate_(self, _note) -> None:
                app.stop()

        self._terminate_observer = _Observer.alloc().init()
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self._terminate_observer, objc.selector(self._terminate_observer.onTerminate_, signature=b"v@:@"), NSApplicationWillTerminateNotification, None
        )

    # -- helpers --------------------------------------------------------------

    def _set_title(self, t: str) -> None:
        self.title = t

    def _set_status(self, s: str) -> None:
        self.status_item.title = f"Status: {s}"


def run_menubar(cfg: Config, dry_run: bool, use_gate: bool) -> int:
    VisualActionsApp(cfg, dry_run, use_gate).run()
    return 0


def mode_icon(mode: str, namespace: str | None) -> str:
    if mode == "armed" and namespace == "media":
        return "🎵"
    if mode == "adjust":
        return "🔊"
    return {"idle": "✋", "holding": "⏳", "armed": "🟢", "dragging": "🤏", "repeat": "🔁"}.get(mode, "✋")
