"""The one place coordinates are flipped and the one feature vector everyone shares.

User frame: +x is the user's right. A laptop camera delivers a raw frame where the
user's right appears on the image's left, so `to_user_frame(mirror=True)` flips x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import INDEX_TIP, MIDDLE_MCP, WRIST, Hand, HandFrame, Landmark

N_FEATURES = 65  # 21 * 3 canonical coords + cos/sin of the index direction


def to_user_frame(hf: HandFrame, mirror: bool) -> HandFrame:
    if not mirror:
        return hf
    return HandFrame(
        t_ns=hf.t_ns,
        hand=hf.hand,
        landmarks=tuple(Landmark(1.0 - lm.x, lm.y, lm.z) for lm in hf.landmarks),
        confidence=hf.confidence,
    )


@dataclass(frozen=True)
class Canonical:
    """Landmarks translated to the wrist, scaled by wrist->middle MCP, left hands mirrored.

    Not rotated, so direction rules (pointing left) can read it directly.
    `pts` is (21, 3) in user-frame orientation (+x user's right, +y down).
    """

    pts: np.ndarray
    scale: float  # original wrist->middle MCP length in frame units
    hand: Hand


def canonical(hf: HandFrame) -> Canonical:
    pts = np.array([(lm.x, lm.y, lm.z) for lm in hf.landmarks], dtype=np.float64)
    pts = pts - pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP][:2]))
    scale = max(scale, 1e-6)
    pts = pts / scale
    if hf.hand is Hand.LEFT:
        pts[:, 0] = -pts[:, 0]
    return Canonical(pts=pts, scale=scale, hand=hf.hand)


def features(hf: HandFrame) -> np.ndarray:
    """Rotation-invariant 63 floats plus the pre-rotation index direction as (cos, sin)."""
    c = canonical(hf)
    pts = c.pts.copy()
    v = pts[INDEX_TIP][:2]
    angle = math.atan2(v[1], v[0])
    up = pts[MIDDLE_MCP][:2]
    rot = -math.atan2(up[0], -up[1])  # rotate so wrist->middle MCP points up (-y)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    xy = pts[:, :2] @ np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    pts[:, :2] = xy
    return np.concatenate([pts.ravel(), [math.cos(angle), math.sin(angle)]]).astype(np.float32)


def wrist_px(hf: HandFrame, width_px: int) -> tuple[float, float]:
    w = hf.landmarks[WRIST]
    return (w.x * width_px, w.y * width_px)
