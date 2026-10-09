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
    elif name == "point_up":
        thumb = THUMB_TUCKED
        fingers = {"index": (up, True), "middle": (up, False), "ring": (up, False), "pinky": (up, False)}
    elif name == "two_up":
        thumb = THUMB_TUCKED
        fingers = {"index": (up, True), "middle": (up, True), "ring": (up, False), "pinky": (up, False)}
    elif name == "thumbs_up":  # fist with a straight thumb pointing up, tip above every finger joint
        thumb = [(-0.5, -0.3), (-0.55, -0.85), (-0.6, -1.35), (-0.62, -1.85)]
        fingers = {f: (up, False) for f in MCPS}
    elif name == "thumbs_down":  # fist with a straight thumb pointing down, tip below every finger joint
        thumb = [(-0.5, -0.3), (-0.6, 0.05), (-0.62, 0.4), (-0.63, 0.75)]
        fingers = {f: (up, False) for f in MCPS}
    elif name == "middle_up":
        thumb = THUMB_TUCKED
        fingers = {"index": (up, False), "middle": (up, True), "ring": (up, False), "pinky": (up, False)}
    elif name == "pinch":  # thumb tip meets a curled index tip, other fingers relaxed-curled
        thumb = [(-0.45, -0.35), (-0.55, -0.75), (-0.45, -1.15), (-0.30, -1.45)]
        index_curl = [MCPS["index"], (-0.45, -1.42), (-0.40, -1.58), (-0.28, -1.43)]
        fingers = {"index": index_curl, "middle": (up, False), "ring": (up, False), "pinky": (up, False)}
    elif name == "none":  # index and pinky out, nothing we bind
        thumb = THUMB_OUT
        fingers = {"index": (up, True), "middle": (up, False), "ring": (up, False), "pinky": (up, True)}
    else:
        raise ValueError(name)
    pts: list[tuple[float, float]] = [(0.0, 0.0)] + thumb
    for f in ("index", "middle", "ring", "pinky"):
        spec = fingers[f]
        if isinstance(spec, list):  # explicit 4 joints
            pts += spec
            continue
        d, ext = spec
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


def drag_frames(
    start_t: float,
    path_points: list[tuple[float, float]],
    seconds: float,
    fps: float = 30.0,
    scale: float = 0.12,
    jitter: float = 0.0005,
    rng: random.Random | None = None,
) -> list[HandFrame]:
    """A pinched hand whose centre moves along `path_points` (user-frame 0..1) over `seconds`."""
    rng = rng or random.Random(2)
    n = max(2, int(seconds * fps))
    out = []
    for i in range(n):
        u = i / (n - 1) * (len(path_points) - 1)
        k = min(int(u), len(path_points) - 2)
        f = u - k
        (x0, y0), (x1, y1) = path_points[k], path_points[k + 1]
        c = (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
        out.append(hand_frame("pinch", int((start_t + i / fps) * 1e9), center=c, scale=scale, jitter=jitter, rng=rng))
    return out


def write_drag_session(
    path: Path,
    palm_seconds: float = 2.5,  # the shipped hold is 1.5 s still palm after the 250 ms token window
    drag_path: list[tuple[float, float]] | None = None,
    drag_seconds: float = 1.0,
    palm_center: tuple[float, float] = (0.5, 0.5),
    fps: float = 30.0,
) -> None:
    """Leader palm, then a pinch-drag along drag_path (default: palm centre -> +0.2 x, +0.1 y), then release."""
    rec = Recorder(path)
    rng = random.Random(3)
    t = 0.0
    for _ in range(int(palm_seconds * fps)):
        rec.write(hand_frame("open_palm", int(t * 1e9), center=palm_center, jitter=0.0005, rng=rng))
        t += 1 / fps
    pts = drag_path or [palm_center, (palm_center[0] + 0.2, palm_center[1] + 0.1)]
    for hf in drag_frames(t, pts, drag_seconds, fps=fps, rng=rng):
        rec.write(hf)
    t += drag_seconds
    for _ in range(int(0.4 * fps)):  # release: open palm again, then lost
        rec.write(hand_frame("open_palm", int(t * 1e9), center=pts[-1], jitter=0.0005, rng=rng))
        t += 1 / fps
    rec.write_lost(int(t * 1e9))
    rec.close()


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
