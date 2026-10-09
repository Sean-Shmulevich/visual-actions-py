"""Typed view of a session's events.log.

Line format (session.py): `elapsed_s  HH:MM:SS.ffffff  kind  text`. Elapsed seconds count
from the recorder's start on the camera's monotonic clock; landmarks.jsonl carries the
raw monotonic t_ns of the same clock, so `t0_ns` below joins the two.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_LINE = re.compile(r"^\s*(-?[\d.]+)\s+(\S+)\s+(\w+)\s+(.*?)\s*$")
_CONF = re.compile(r"conf=([\d.]+)")
_MODE = re.compile(r"^(\w+) -> (\w+)(?: \[(\w+)\])?")


@dataclass(frozen=True)
class Event:
    t: float  # elapsed seconds
    wall: str
    kind: str  # token | mode | hand | action | drag | snap | veto | session
    text: str

    # -- convenience views ------------------------------------------------------
    @property
    def token(self) -> str | None:
        return self.text.split(" ", 1)[0] if self.kind == "token" else None

    @property
    def confidence(self) -> float | None:
        m = _CONF.search(self.text)
        return float(m.group(1)) if m else None

    @property
    def still(self) -> bool | None:
        if self.kind != "token":
            return None
        return "still=True" in self.text

    @property
    def mode_from(self) -> str | None:
        m = _MODE.match(self.text) if self.kind == "mode" else None
        return m.group(1) if m else None

    @property
    def mode_to(self) -> str | None:
        m = _MODE.match(self.text) if self.kind == "mode" else None
        return m.group(2) if m else None

    @property
    def namespace(self) -> str | None:
        m = _MODE.match(self.text) if self.kind == "mode" else None
        return m.group(3) if m else None

    @property
    def action_name(self) -> str | None:
        if self.kind != "action":
            return None
        head, _, tail = self.text.rpartition(" ")
        return head if tail == "ok" else self.text.split(" FAILED")[0]

    @property
    def action_ok(self) -> bool | None:
        return self.text.endswith(" ok") if self.kind == "action" else None

    @property
    def hand_seen(self) -> bool | None:
        if self.kind != "hand":
            return None
        return self.text.startswith("seen")

    @property
    def drag_phase(self) -> str | None:
        return self.text.split(" ", 1)[0] if self.kind == "drag" else None


def parse_events(path: Path) -> list[Event]:
    out: list[Event] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LINE.match(line)
            if m:
                out.append(Event(float(m.group(1)), m.group(2), m.group(3), m.group(4)))
    return out


def session_t0_ns(events: list[Event], landmarks: Path) -> int | None:
    """The monotonic t_ns that elapsed 0.0 corresponds to.

    Newer sessions log it on the `session` line (`t0_ns=...`). Older ones are joined by the
    first `hand seen` line and the first landmark frame, which are the same camera frame.
    """
    for ev in events:
        if ev.kind == "session":
            m = re.search(r"t0_ns=(\d+)", ev.text)
            if m:
                return int(m.group(1))
    first_seen = next((ev for ev in events if ev.kind == "hand" and ev.hand_seen), None)
    if first_seen is None or not landmarks.exists():
        return None
    with landmarks.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith('{"t_ns"'):
                t_ns = int(json.loads(line)["t_ns"])
                return t_ns - int(first_seen.t * 1e9)
    return None


def session_duration_s(events: list[Event]) -> float:
    return events[-1].t if events else 0.0
