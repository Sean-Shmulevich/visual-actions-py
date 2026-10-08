"""Audio feedback: a sound per state change, played off the main thread via afplay.

Interim until the overlay exists; stays as the `audio = true` feature after that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..core.drag import DragEvent, DragPhase
from ..core.events import ActionFired, Bus, ModeChanged
from ..core.modes import ADJUST, ARMED, DRAGGING, HOLDING, IDLE

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
        self._mode = IDLE

    def _on_mode(self, ev: ModeChanged) -> None:
        self._mode = ev.new
        if ev.new in (DRAGGING, ADJUST):
            play("grab")
        elif ev.old in (DRAGGING, ADJUST) and ev.new in (IDLE, ARMED):
            play("drop")
        elif ev.new == ARMED and ev.old == HOLDING:
            play("armed")  # only on opening a menu, not on returning to it after each chained command
        elif ev.old == ARMED and ev.new == IDLE and self._fired_at != ev.t_ns:
            play("timeout")  # left ARMED without an action at this instant
        elif ev.old == HOLDING and ev.new == IDLE:
            pass  # hold broken: silent, it happens constantly

    def _on_action(self, ev: ActionFired) -> None:
        self._fired_at = ev.t_ns
        if self._mode == ADJUST and ev.ok:
            return  # volume steps while pinching: the system HUD is the feedback, a chime per step is noise
        play("fired" if ev.ok else "failed")
