"""Synthetic hand poses for fixtures and tests. Geometry in canonical units
(wrist at origin, wrist->middle MCP = 1, +x user's right, +y down), placed into a
raw camera frame (mirrored) so fixtures look like real recordings."""

from __future__ import annotations

import math
import random
from pathlib import Path

from ..core.recorder import Recorder
from ..core.types import Hand, HandFrame, Landmark

MCPS = {"index": (-0.30, -1.00), "middle": (0.0, -1.0), "ring": (0.28, -0.95), "pinky": (0.55, -0.85)}
THUMB_OUT = [(-0.5, -0.3), (-0.9, -0.6), (-1.2, -0.8), (-1.5, -1.0)]
THUMB_TUCKED = [(-0.4, -0.4), (-0.3, -0.6), (-0.1, -0.7), (0.1, -0.75)]


def _finger(mcp: tuple[float, float], direction: tuple[float, float], extended: bool) -> list[tuple[float, float]]:
    dx, dy = direction
    if extended:
        return [mcp, (mcp[0] + dx * 0.45, mcp[1] + dy * 0.45), (mcp[0] + dx * 0.75, mcp[1] + dy * 0.75), (mcp[0] + dx, mcp[1] + dy)]
    # curled: pip up, dip back down, tip beside the mcp
    return [mcp, (mcp[0] + dx * 0.45, mcp[1] + dy * 0.45), (mcp[0] + dx * 0.35 + 0.1, mcp[1] + dy * 0.1), (mcp[0] + 0.08, mcp[1] + 0.05)]


def pose(name: str) -> list[tuple[float, float]]:
    up = (0.0, -1.0)
    left = (-1.0, 0.0)
    right = (1.0, 0.0)
    spread = {"index": (-0.15, -1.0), "middle": up, "ring": (0.15, -1.0), "pinky": (0.3, -0.95)}
    if name == "open_palm":
        thumb, fingers = THUMB_OUT, {f: (spread[f], True) for f in MCPS}
    elif name == "fist":
        thumb, fingers = THUMB_TUCKED, {f: (up, False) for f in MCPS}
    elif name == "h_left":
        thumb = THUMB_TUCKED
        fingers = {"index": (left, True), "middle": (left, True), "ring": (up, False), "pinky": (up, False)}
    elif name == "h_right":
        thumb = THUMB_TUCKED
        fingers = {"index": (right, True), "middle": (right, True), "ring": (up, False), "pinky": (up, False)}
    elif name == "none":  # index only, pointing up
        thumb = THUMB_OUT
        fingers = {"index": (up, True), "middle": (up, False), "ring": (up, False), "pinky": (up, False)}
    else:
        raise ValueError(name)
    pts: list[tuple[float, float]] = [(0.0, 0.0)] + thumb
    for f in ("index", "middle", "ring", "pinky"):
        d, ext = fingers[f]
        n = math.hypot(*d)
        pts += _finger(MCPS[f], (d[0] / n, d[1] / n), ext)
    return pts


def hand_frame(
    name: str,
    t_ns: int,
    hand: Hand = Hand.RIGHT,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.12,
    jitter: float = 0.0,
    mirror_to_raw: bool = True,
    rng: random.Random | None = None,
) -> HandFrame:
    rng = rng or random.Random(0)
    lms = []
    for x, y in pose(name):
        ux = center[0] + x * scale + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        uy = center[1] + y * scale + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        if hand is Hand.LEFT:
            ux = 2 * center[0] - ux
        rx = 1.0 - ux if mirror_to_raw else ux
        lms.append(Landmark(rx, uy, 0.0))
    return HandFrame(t_ns=t_ns, hand=hand, landmarks=tuple(lms), confidence=0.95)


def write_session(path: Path, segments: list[tuple[str, float]], fps: float = 30.0, jitter: float = 0.0005) -> None:
    """segments: [(pose name or 'lost', seconds), ...]"""
    rec = Recorder(path)
    rng = random.Random(1)
    t = 0.0
    for name, seconds in segments:
        if name == "lost":
            rec.write_lost(int(t * 1e9))
            t += seconds
            continue
        n = int(seconds * fps)
        for _ in range(n):
            rec.write(hand_frame(name, int(t * 1e9), jitter=jitter, rng=rng))
            t += 1 / fps
    rec.close()


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    write_session(out / "h_left.jsonl", [("open_palm", 2.5), ("h_left", 0.8), ("lost", 0.5)])
    write_session(out / "h_right.jsonl", [("open_palm", 2.5), ("h_right", 0.8), ("lost", 0.5)])
    write_session(out / "timeout.jsonl", [("open_palm", 2.5), ("none", 6.0), ("lost", 0.5)])
    write_session(out / "no_leader.jsonl", [("h_left", 2.0), ("lost", 0.5)])
    write_session(out / "escape_fist.jsonl", [("open_palm", 2.5), ("fist", 1.5), ("h_left", 0.8), ("lost", 0.5)])
    print("wrote fixtures to", out)
