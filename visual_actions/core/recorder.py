"""JSONL datasets of raw-frame HandFrames. One line per frame; {"lost": t_ns} marks a hand lost.

A frame line may carry extra keys beside the HandFrame fields (a live session adds
"face", the hand/face box overlap the capture thread computed, so a replay can feed the
palm veto the same number). `from_json` ignores them; `read_records` hands them back.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from .types import Hand, HandFrame, Landmark

Record = HandFrame | int  # int = HandLost at t_ns
FRAME_KEYS = frozenset({"t_ns", "hand", "confidence", "landmarks"})


def to_json(hf: HandFrame, extra: dict[str, Any] | None = None) -> dict:
    d = {
        "t_ns": hf.t_ns,
        "hand": hf.hand.value,
        "confidence": hf.confidence,
        "landmarks": [[lm.x, lm.y, lm.z] for lm in hf.landmarks],
    }
    if extra:
        d.update(extra)
    return d


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
        self._f: IO[str] = path.open("w", encoding="utf-8", buffering=1)  # line-buffered: survives a hard quit
        self.count = 0

    def write(self, hf: HandFrame, extra: dict[str, Any] | None = None) -> None:
        self._f.write(json.dumps(to_json(hf, extra)) + "\n")
        self.count += 1

    def write_lost(self, t_ns: int) -> None:
        self._f.write(json.dumps({"lost": t_ns}) + "\n")

    def close(self) -> None:
        self._f.close()


def read_records(path: Path) -> Iterator[tuple[Record, dict[str, Any]]]:
    """Every record with the extra keys its line carried (empty for lost markers and plain datasets)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "lost" in d:
                yield int(d["lost"]), {}
            else:
                yield from_json(d), {k: v for k, v in d.items() if k not in FRAME_KEYS}


def read_session(path: Path) -> Iterator[Record]:
    for rec, _extra in read_records(path):
        yield rec
