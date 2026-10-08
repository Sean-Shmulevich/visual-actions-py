"""Parse "cmd+shift+tab" into modifiers and a key name. Platform drivers map names to codes."""

from __future__ import annotations

from dataclasses import dataclass

MODIFIERS = {"cmd", "command", "ctrl", "control", "alt", "option", "opt", "shift", "fn"}
CANON = {"command": "cmd", "control": "ctrl", "option": "alt", "opt": "alt"}


@dataclass(frozen=True)
class Chord:
    modifiers: tuple[str, ...]  # canonical: cmd, ctrl, alt, shift, fn; in given order
    key: str  # lowercase key name: "tab", "a", "f11", "left"


def parse_chord(text: str) -> Chord:
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty chord")
    mods: list[str] = []
    for p in parts[:-1]:
        if p not in MODIFIERS:
            raise ValueError(f"unknown modifier {p!r} in chord {text!r}")
        mods.append(CANON.get(p, p))
    key = parts[-1]
    if key in MODIFIERS:
        raise ValueError(f"chord {text!r} has no main key")
    return Chord(tuple(mods), key)
