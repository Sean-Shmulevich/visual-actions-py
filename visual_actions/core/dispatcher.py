"""Action (data) -> DesktopAutomation calls. Always publishes ActionFired."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .automation import DesktopAutomation, MediaVerb
from .events import ActionFired, Bus
from .types import Action, ActionKind

PluginRunner = Callable[[Action], tuple[bool, str]]


class Dispatcher:
    def __init__(self, bus: Bus, automation: DesktopAutomation, plugin_runner: PluginRunner | None = None) -> None:
        self.bus = bus
        self.automation = automation
        self.plugin_runner = plugin_runner

    def dispatch(self, action: Action, t_ns: int) -> None:
        ok, message = True, ""
        try:
            if action.kind is ActionKind.KEY:
                chord = action.arg("chord")
                if not chord:
                    raise ValueError("key action without chord")
                self.automation.press(chord)
            elif action.kind is ActionKind.SCROLL:
                self.automation.scroll(int(action.arg("dx", "0") or 0), int(action.arg("dy", "0") or 0))
            elif action.kind is ActionKind.MEDIA:
                self.automation.media(MediaVerb(action.arg("verb", "play_pause")))
            elif action.kind is ActionKind.PLUGIN:
                if self.plugin_runner is None:
                    raise RuntimeError("no plugin runner configured")
                ok, message = self.plugin_runner(action)
            elif action.kind is ActionKind.WINDOW:
                raise NotImplementedError("window actions arrive with the AX driver")
            else:
                raise ValueError(f"unknown action kind {action.kind}")
        except Exception as exc:  # noqa: BLE001 - the overlay shows failures; the loop must not die
            ok, message = False, f"{type(exc).__name__}: {exc}"
        self.bus.publish(ActionFired(t_ns=t_ns, action=action, ok=ok, message=message))


def script_path_for(action: Action) -> Path | None:
    p = action.arg("script")
    return Path(p) if p else None
