"""Audio feedback: a sound per state change, played off the main thread via afplay.

Interim until the overlay exists; stays as the `audio = true` feature after that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..core.drag import DragEvent, DragPhase
from ..core.events import ActionFired, Bus, ModeChanged
from ..core.modes import ARMED, DRAGGING, HOLDING, IDLE

SOUNDS = Path("/System/Library/Sounds")
MAP = {
    "armed": SOUNDS / "Tink.aiff",
    "fired": SOUNDS / "Pop.aiff",
    "failed": SOUNDS / "Basso.aiff",
    "timeout": SOUNDS / "Bottle.aiff",
    "grab": SOUNDS / "Morse.aiff",
    "drop": SOUNDS / "Pop.aiff",
    "lost": SOUNDS / "Sosumi.aiff",
}


def play(name: str) -> None:
    p = MAP.get(name)
    if p is None or sys.platform != "darwin" or not p.exists():
        return
    subprocess.Popen(["afplay", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SoundFeedback:
    def __init__(self, bus: Bus) -> None:
        bus.subscribe(ModeChanged, self._on_mode)
        bus.subscribe(ActionFired, self._on_action)
        bus.subscribe(DragEvent, lambda e: play("lost") if e.phase is DragPhase.PAUSE else None)
        self._fired_at: int | None = None

    def _on_mode(self, ev: ModeChanged) -> None:
        if ev.new == DRAGGING:
            play("grab")
        elif ev.old == DRAGGING and ev.new == IDLE:
            play("drop")
        elif ev.new == ARMED:
            play("armed")
        elif ev.old == ARMED and ev.new == IDLE and self._fired_at != ev.t_ns:
            play("timeout")  # left ARMED without an action at this instant
        elif ev.old == HOLDING and ev.new == IDLE:
            pass  # hold broken: silent, it happens constantly

    def _on_action(self, ev: ActionFired) -> None:
        self._fired_at = ev.t_ns
        play("fired" if ev.ok else "failed")
