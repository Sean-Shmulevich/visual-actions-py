"""JSONL datasets of raw-frame HandFrames. One line per frame; {"lost": t_ns} marks a hand lost."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from .types import Hand, HandFrame, Landmark

Record = HandFrame | int  # int = HandLost at t_ns


def to_json(hf: HandFrame) -> dict:
    return {
        "t_ns": hf.t_ns,
        "hand": hf.hand.value,
        "confidence": hf.confidence,
        "landmarks": [[lm.x, lm.y, lm.z] for lm in hf.landmarks],
    }


def from_json(d: dict) -> HandFrame:
    return HandFrame(
        t_ns=int(d["t_ns"]),
        hand=Hand(d["hand"]),
        landmarks=tuple(Landmark(float(x), float(y), float(z)) for x, y, z in d["landmarks"]),
        confidence=float(d.get("confidence", 1.0)),
    )


class Recorder:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f: IO[str] = path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, hf: HandFrame) -> None:
        self._f.write(json.dumps(to_json(hf)) + "\n")
        self.count += 1

    def write_lost(self, t_ns: int) -> None:
        self._f.write(json.dumps({"lost": t_ns}) + "\n")

    def close(self) -> None:
        self._f.close()


def read_session(path: Path) -> Iterator[Record]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "lost" in d:
                yield int(d["lost"])
            else:
                yield from_json(d)
