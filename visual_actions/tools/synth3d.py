"""Kinematic 3D hand: unlimited labelled MediaPipe-style landmarks for training and tests.

A parametric right hand (left = mirror) is posed by joint angles, rotated, placed in a
640x480 webcam frame with perspective, jittered, and written in the RAW camera frame
(user's right appears on the image's left), so the pipeline's mirror step applies
exactly as with real recordings.

Segment proportions follow Buchholz, Armstrong & Goldstein, "Anthropometric data for
describing the kinematics of the human hand", Ergonomics 35(3), 1992 (segment lengths
as fractions of hand length; e.g. middle finger proximal:middle:distal = 0.277:0.170:
0.108, index 0.265:0.143:0.097, ring 0.259:0.165:0.107, little 0.206:0.117:0.093,
thumb metacarpal:proximal:distal = 0.251:0.196:0.158). MediaPipe's wrist landmark sits
at the wrist crease and its MCPs are nearly equidistant from it, so the palm is
calibrated to MediaPipe geometry measured on real recordings (wrist->middle MCP = 1;
finger lengths 0.74 / 0.86 / 0.81 / 0.67 of that for index / middle / ring / little)
while the within-finger proportions are Buchholz's.

Local frame (right hand, palm toward the camera, as the user sees it in a mirror):
+x toward the little finger (user's right), +y along the middle metacarpal (up),
+z out of the palm toward the camera. Flexion bends a finger from +y toward +z.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.recorder import Recorder
from ..core.types import Hand, HandFrame, Landmark

FINGERS = ("index", "middle", "ring", "pinky")

# Palm: MCP position relative to the wrist as (angle from +y toward +x, length) and a
# little arch in z. Calibrated on real open-palm recordings.
MCP_POLAR = {
    "index": (-9.0, 0.97),
    "middle": (0.0, 1.00),
    "ring": (9.0, 0.96),
    "pinky": (19.0, 0.89),
}
MCP_Z = {"index": 0.00, "middle": 0.02, "ring": 0.05, "pinky": 0.09}
# Phalanx lengths (proximal, middle, distal) in units of wrist->middle MCP.
PHALANX = {
    "index": (0.39, 0.21, 0.14),
    "middle": (0.43, 0.26, 0.17),
    "ring": (0.395, 0.25, 0.165),
    "pinky": (0.33, 0.19, 0.15),
}
THUMB_CMC = np.array([-0.26, 0.15, 0.06])
THUMB_SEG = (0.36, 0.28, 0.22)  # metacarpal, proximal, distal
THUMB_DIR0 = np.array([-0.62, 0.72, 0.30])  # metacarpal direction, abducted open palm
THUMB_OPP = np.array([0.80, -0.25, 0.55])  # direction the thumb sweeps toward when flexing


@dataclass
class FingerPose:
    mcp: float = 0.0  # flexion, degrees
    pip: float = 0.0
    abd: float = 0.0  # abduction toward +x (little-finger side), degrees
    dip_ratio: float = 0.667


@dataclass
class ThumbPose:
    flex: float = 0.0  # CMC sweep toward the palm (opposition), degrees
    abd: float = 30.0  # in-plane spread about +z, degrees (positive = away from the palm)
    mcp: float = 0.0
    ip: float = 0.0


@dataclass
class HandPose:
    fingers: dict[str, FingerPose] = field(
        default_factory=lambda: {f: FingerPose() for f in FINGERS}
    )
    thumb: ThumbPose = field(default_factory=ThumbPose)
    hand: Hand = Hand.RIGHT
    sideways: bool = False  # hint for 'none': render with a sideways (H-sign) view regime


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = _unit(axis)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y, z = a
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def forward_kinematics(pose: HandPose) -> np.ndarray:
    """21 x 3 local points in MediaPipe order. Right hand; a left hand is mirrored in x."""
    pts = np.zeros((21, 3))
    palm_n = np.array([0.0, 0.0, 1.0])
    # thumb
    d = THUMB_DIR0.copy()
    d = _rot(palm_n, -pose.thumb.abd + 30.0) @ d  # abd=30 is the calibrated open-palm spread
    flex_axis = _unit(np.cross(d, THUMB_OPP))
    seg = [_rot(flex_axis, pose.thumb.flex) @ d]
    seg.append(_rot(flex_axis, pose.thumb.flex + pose.thumb.mcp) @ d)
    seg.append(_rot(flex_axis, pose.thumb.flex + pose.thumb.mcp + pose.thumb.ip) @ d)
    p = THUMB_CMC.copy()
    pts[1] = p
    for i, (v, L) in enumerate(zip(seg, THUMB_SEG)):
        p = p + _unit(v) * L
        pts[2 + i] = p
    # fingers
    for fi, name in enumerate(FINGERS):
        ang, length = MCP_POLAR[name]
        d0 = np.array([math.sin(math.radians(ang)), math.cos(math.radians(ang)), 0.0])
        mcp = d0 * length + np.array([0.0, 0.0, MCP_Z[name]])
        fp = pose.fingers[name]
        d = _rot(palm_n, -fp.abd) @ d0  # abduction toward +x is a negative rotation about +z
        side = _unit(
            np.cross(palm_n, d)
        )  # flexion axis: rotating d about `side` moves it toward +z
        angles = (fp.mcp, fp.mcp + fp.pip, fp.mcp + fp.pip + fp.pip * fp.dip_ratio)
        base = 5 + 4 * fi
        pts[base] = mcp
        p = mcp
        for k, (a, L) in enumerate(zip(angles, PHALANX[name])):
            p = p + (_rot(side, -a) @ d) * L
            pts[base + 1 + k] = p
    if pose.hand is Hand.LEFT:
        pts[:, 0] = -pts[:, 0]
    return pts


# --------------------------------------------------------------------------- poses


def _u(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def pose_open_palm(rng: random.Random) -> HandPose:
    p = HandPose()
    spread = {
        "index": _u(rng, -16, -5),
        "middle": _u(rng, -3, 3),
        "ring": _u(rng, 3, 12),
        "pinky": _u(rng, 8, 24),
    }
    for f in FINGERS:
        p.fingers[f] = FingerPose(mcp=_u(rng, -12, 18), pip=_u(rng, 0, 18), abd=spread[f])
    p.thumb = ThumbPose(
        flex=_u(rng, 0, 18), abd=_u(rng, 20, 50), mcp=_u(rng, 0, 12), ip=_u(rng, 0, 12)
    )
    return p


def pose_fist(rng: random.Random) -> HandPose:
    p = HandPose()
    for f in FINGERS:
        p.fingers[f] = FingerPose(mcp=_u(rng, 65, 100), pip=_u(rng, 85, 125), abd=_u(rng, -4, 4))
    if rng.random() < 0.5:  # thumb wrapped across the fingers
        p.thumb = ThumbPose(
            flex=_u(rng, 35, 75), abd=_u(rng, -5, 15), mcp=_u(rng, 25, 60), ip=_u(rng, 10, 50)
        )
    else:  # thumb lying along the index, pointing up
        p.thumb = ThumbPose(
            flex=_u(rng, 10, 35), abd=_u(rng, 0, 20), mcp=_u(rng, 15, 45), ip=_u(rng, 5, 35)
        )
    return p


def _two_extended(rng: random.Random) -> HandPose:
    p = HandPose()
    p.fingers["index"] = FingerPose(mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, -4, 4))
    p.fingers["middle"] = FingerPose(mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, -6, 2))
    p.fingers["ring"] = FingerPose(mcp=_u(rng, 55, 100), pip=_u(rng, 70, 125), abd=_u(rng, -3, 5))
    p.fingers["pinky"] = FingerPose(mcp=_u(rng, 55, 100), pip=_u(rng, 70, 125), abd=_u(rng, -3, 8))
    p.thumb = ThumbPose(
        flex=_u(rng, 45, 80), abd=_u(rng, -5, 15), mcp=_u(rng, 20, 55), ip=_u(rng, 10, 45)
    )
    return p


def pose_h(rng: random.Random) -> HandPose:
    p = _two_extended(rng)
    if rng.random() < 0.5:  # a thumb hidden behind the palm is estimated lying along the index
        p.thumb = ThumbPose(
            flex=_u(rng, 15, 40), abd=_u(rng, -5, 15), mcp=_u(rng, 10, 40), ip=_u(rng, 0, 30)
        )
    return p


def _thumb_along_index(rng: random.Random) -> ThumbPose:
    """Thumb lying along the index, tip near the index PIP: how a tucked or hidden thumb
    is usually estimated, and how people actually hold point_up / two_up."""
    return ThumbPose(
        flex=_u(rng, 12, 38), abd=_u(rng, -5, 15), mcp=_u(rng, 10, 40), ip=_u(rng, 0, 30)
    )


def pose_two_up(rng: random.Random) -> HandPose:
    p = _two_extended(rng)
    if rng.random() < 0.6:
        p.thumb = _thumb_along_index(rng)
    return p


def pose_point_up(rng: random.Random) -> HandPose:
    p = HandPose()
    p.fingers["index"] = FingerPose(mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, -6, 6))
    for f in ("middle", "ring", "pinky"):
        p.fingers[f] = FingerPose(mcp=_u(rng, 55, 100), pip=_u(rng, 70, 125), abd=_u(rng, -3, 5))
    if rng.random() < 0.4:
        p.thumb = ThumbPose(
            flex=_u(rng, 45, 80), abd=_u(rng, -5, 15), mcp=_u(rng, 20, 55), ip=_u(rng, 10, 45)
        )
    else:
        p.thumb = _thumb_along_index(rng)
    return p


def _solve_pinch(p: HandPose, rng: random.Random, iters: int = 120) -> HandPose:
    """Random search with restarts on the thumb so the thumb tip meets the index tip."""
    starts = (
        ThumbPose(55, 5, 25, 15),
        ThumbPose(85, 25, 45, 35),
        ThumbPose(40, -15, 15, 10),
        ThumbPose(100, 40, 55, 45),
    )
    best, best_d = p.thumb, float("inf")
    for start in starts:
        base, base_d = start, float("inf")
        for i in range(iters // len(starts) + 1):
            spread = 35.0 if i < 10 else 12.0 if i < 22 else 5.0
            cand = ThumbPose(
                flex=max(0, min(125, base.flex + _u(rng, -spread, spread))),
                abd=max(-35, min(55, base.abd + _u(rng, -spread, spread))),
                mcp=max(0, min(85, base.mcp + _u(rng, -spread, spread))),
                ip=max(0, min(85, base.ip + _u(rng, -spread, spread))),
            )
            p.thumb = cand
            pts = forward_kinematics(p)
            d = float(np.linalg.norm(pts[4] - pts[8]))
            if d < base_d:
                base, base_d = cand, d
        if base_d < best_d:
            best, best_d = base, base_d
        if best_d < 0.08:
            break
    p.thumb = best
    return p


def pose_pinch(rng: random.Random) -> HandPose:
    p = HandPose()
    p.fingers["index"] = FingerPose(mcp=_u(rng, 35, 70), pip=_u(rng, 35, 80), abd=_u(rng, -10, 2))
    for f in ("middle", "ring", "pinky"):
        if rng.random() < 0.5:
            p.fingers[f] = FingerPose(
                mcp=_u(rng, 10, 45), pip=_u(rng, 10, 50), abd=_u(rng, -3, 8)
            )  # relaxed
        else:
            p.fingers[f] = FingerPose(
                mcp=_u(rng, 50, 95), pip=_u(rng, 60, 120), abd=_u(rng, -3, 6)
            )  # curled
    return _solve_pinch(p, rng)


def _finger_state(fp: FingerPose) -> str:
    if fp.mcp < 25 and fp.pip < 30:
        return "E"
    if fp.mcp > 50 and fp.pip > 65:
        return "C"
    return "?"


BOUND_PATTERNS = {"EEEE", "CCCC", "EECC", "ECCC"}


def pose_none(rng: random.Random) -> HandPose:
    """Plausible poses that are none of the bound gestures, near-misses included."""
    while True:
        p = HandPose()
        kind = rng.random()
        if kind < 0.25:  # independent random fingers
            for f in FINGERS:
                mode = rng.random()
                if mode < 0.33:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, -10, 25), pip=_u(rng, 0, 30), abd=_u(rng, -8, 8)
                    )
                elif mode < 0.66:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, 25, 55), pip=_u(rng, 25, 70), abd=_u(rng, -5, 5)
                    )
                else:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, 55, 100), pip=_u(rng, 65, 125), abd=_u(rng, -4, 4)
                    )
        elif kind < 0.45:  # relaxed hand: fingers together and slightly bent, or half-open
            if rng.random() < 0.5:
                for f in FINGERS:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, 12, 35), pip=_u(rng, 15, 45), abd=_u(rng, -3, 3)
                    )
            else:
                for f in FINGERS:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, 20, 50), pip=_u(rng, 20, 60), abd=_u(rng, -6, 6)
                    )
        elif kind < 0.6:  # index + pinky ("rock"), or three fingers
            ext = {"index", "pinky"} if rng.random() < 0.5 else {"index", "middle", "ring"}
            for f in FINGERS:
                if f in ext:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, -6, 6)
                    )
                else:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, 55, 100), pip=_u(rng, 70, 125), abd=_u(rng, -3, 5)
                    )
        elif kind < 0.75:  # thumbs up / middle finger only / pinky only
            only = rng.choice(["middle", "pinky", "ring", None])
            for f in FINGERS:
                if f == only:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, -6, 6)
                    )
                else:
                    p.fingers[f] = FingerPose(
                        mcp=_u(rng, 55, 100), pip=_u(rng, 70, 125), abd=_u(rng, -3, 5)
                    )
            if only is None:
                p.thumb = ThumbPose(
                    flex=_u(rng, -10, 10),
                    abd=_u(rng, 40, 70),
                    mcp=_u(rng, -5, 10),
                    ip=_u(rng, 0, 10),
                )
        else:  # V sign: index + middle spread apart, thumb out; the classic near-miss of the H sign
            p.fingers["index"] = FingerPose(
                mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, -25, -10)
            )
            p.fingers["middle"] = FingerPose(
                mcp=_u(rng, -8, 18), pip=_u(rng, 0, 15), abd=_u(rng, 8, 22)
            )
            for f in ("ring", "pinky"):
                p.fingers[f] = FingerPose(
                    mcp=_u(rng, 30, 100), pip=_u(rng, 40, 125), abd=_u(rng, -3, 5)
                )
            p.thumb = ThumbPose(
                flex=_u(rng, 0, 25), abd=_u(rng, 25, 60), mcp=_u(rng, 0, 20), ip=_u(rng, 0, 20)
            )
            p.sideways = rng.random() < NONE_SIDEWAYS_P
        if p.thumb == ThumbPose():
            p.thumb = ThumbPose(
                flex=_u(rng, 0, 70), abd=_u(rng, -5, 50), mcp=_u(rng, 0, 50), ip=_u(rng, 0, 40)
            )
        pattern = "".join(_finger_state(p.fingers[f]) for f in FINGERS)
        if pattern in BOUND_PATTERNS:
            continue
        spread_v = pattern == "EE??" or pattern == "EEC?" or pattern == "EE?C"
        if spread_v and abs(p.fingers["index"].abd - p.fingers["middle"].abd) < 15:
            continue
        return p


# --------------------------------------------------------------------------- rendering


@dataclass
class View:
    yaw: float = 0.0  # about +y (vertical), degrees
    pitch: float = 0.0  # about +x, degrees
    roll: float = 0.0  # about +z (in-plane), degrees; positive turns fingers toward the user's left
    scale: float = 0.3  # wrist->middle MCP in image-width units at the hand's depth
    center: tuple[float, float] = (0.5, 0.5)  # user-frame image position of the wrist
    depth: float = 12.0  # camera distance in palm units (perspective strength)
    z_gain: float = 0.8  # MediaPipe-like z scale relative to x
    jitter: float = 0.005  # Gaussian noise as a fraction of hand size
    palm_scale: float = (
        1.0  # shrinks wrist->MCP: MediaPipe's wrist drifts toward the palm on some views
    )


@dataclass
class ViewPreset:
    """Per-class orientation distribution. Angles in degrees; ranges are half-widths.

    Pitch is positive when the fingers tilt toward the camera; a laptop camera at eye
    level sees a raised hand that way, so most classes lean positive.
    """

    yaw_c: float = 0.0
    yaw_r: float = 35.0
    pitch_c: float = 20.0
    pitch_r: float = 30.0
    roll_c: float = 0.0
    roll_r: float = 25.0
    palm: tuple[float, float] = (0.75, 1.05)
    depth: tuple[float, float] = (7.0, 20.0)
    z_gain: tuple[float, float] = (0.5, 1.0)
    flip_yaw_p: float = 0.0  # probability of showing the back of the hand (yaw + 180)


# Calibrated 2026-10-08 by random search against real recordings (feature-space centroid
# and spread), then rounded and widened so no range is narrower than a casual user's
# variation. Re-run the calibration when real data from more people arrives.
VIEW_PRESETS: dict[str, ViewPreset] = {
    "open_palm": ViewPreset(
        yaw_c=-10,
        yaw_r=35,
        pitch_c=5,
        pitch_r=45,
        roll_c=5,
        roll_r=15,
        palm=(0.9, 1.15),
        depth=(10, 30),
        z_gain=(0.3, 0.75),
    ),
    "fist": ViewPreset(
        yaw_c=-40,
        yaw_r=60,
        pitch_c=40,
        pitch_r=70,
        roll_c=15,
        roll_r=35,
        palm=(1.0, 1.2),
        depth=(4, 20),
        z_gain=(0.35, 0.9),
    ),
    "h_left": ViewPreset(
        yaw_c=200,
        yaw_r=25,
        pitch_c=-50,
        pitch_r=35,
        roll_c=97,
        roll_r=20,
        palm=(0.7, 1.15),
        depth=(8, 24),
        z_gain=(0.4, 1.05),
        flip_yaw_p=0.05,
    ),
    "h_right": ViewPreset(
        yaw_c=20,
        yaw_r=25,
        pitch_c=-43,
        pitch_r=25,
        roll_c=-83,
        roll_r=30,
        palm=(0.65, 1.15),
        depth=(10, 25),
        z_gain=(0.25, 0.85),
        flip_yaw_p=0.15,
    ),
    "point_up": ViewPreset(
        yaw_c=-37,
        yaw_r=50,
        pitch_c=13,
        pitch_r=25,
        roll_c=-20,
        roll_r=15,
        palm=(1.0, 1.2),
        depth=(7, 28),
        z_gain=(0.35, 0.85),
    ),
    "two_up": ViewPreset(
        yaw_c=-32,
        yaw_r=45,
        pitch_c=11,
        pitch_r=30,
        roll_c=-27,
        roll_r=15,
        palm=(0.95, 1.2),
        depth=(8, 24),
        z_gain=(0.5, 0.9),
    ),
    "pinch": ViewPreset(
        yaw_c=-20,
        yaw_r=50,
        pitch_c=20,
        pitch_r=45,
        roll_c=0,
        roll_r=35,
        palm=(0.8, 1.15),
        depth=(6, 24),
        z_gain=(0.4, 0.9),
    ),
    "none": ViewPreset(
        yaw_c=-27,
        yaw_r=65,
        pitch_c=50,
        pitch_r=80,
        roll_c=-15,
        roll_r=75,
        palm=(0.7, 1.2),
        depth=(6, 18),
        z_gain=(0.6, 0.85),
    ),
}


def random_view(rng: random.Random, cls: str, preset: ViewPreset | None = None) -> View:
    pr = preset or VIEW_PRESETS.get(cls, ViewPreset())
    yaw = pr.yaw_c + _u(rng, -pr.yaw_r, pr.yaw_r)
    if rng.random() < pr.flip_yaw_p:
        yaw += 180.0
    scale = math.exp(_u(rng, math.log(0.09), math.log(SCALE_MAX)))
    if KEEP_IN_FRAME:
        half = min(0.45, 1.0 * scale)  # a hand spans ~2 palm units; keep it inside the frame
        center = (_u(rng, 0.05 + half, 0.95 - half), _u(rng, 0.1 + half, 0.95 - half))
    else:  # real webcams cut hands off at the edge all the time; MediaPipe still reports them
        center = (_u(rng, 0.25, 0.75), _u(rng, 0.3, 0.75))
    return View(
        yaw=yaw,
        pitch=pr.pitch_c + _u(rng, -pr.pitch_r, pr.pitch_r),
        roll=pr.roll_c + _u(rng, -pr.roll_r, pr.roll_r),
        scale=scale,
        center=center,
        depth=_u(rng, *pr.depth),
        z_gain=_u(rng, *pr.z_gain),
        jitter=_u(rng, 0.003, 0.01),
        palm_scale=_u(rng, *pr.palm),
    )


def render(pts_local: np.ndarray, view: View, rng: random.Random | None = None) -> np.ndarray:
    """Local 21x3 -> raw-camera-frame landmarks (x, y in 0..1, z MediaPipe-relative)."""
    R = (
        _rot(np.array([0, 0, 1.0]), view.roll)
        @ _rot(np.array([0, 1.0, 0]), view.yaw)
        @ _rot(np.array([1.0, 0, 0]), view.pitch)
    )
    local = pts_local.copy()
    if view.palm_scale != 1.0:  # move the wrist toward the palm centre, keeping the fingers
        palm_centre = local[[5, 9, 13, 17]].mean(axis=0)
        local[0] = palm_centre + (local[0] - palm_centre) * view.palm_scale
        local[1] = palm_centre + (local[1] - palm_centre) * (0.5 + 0.5 * view.palm_scale)
    p = local @ R.T
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    f = (
        view.scale * view.depth
    )  # focal length so that wrist->MCP projects to `scale` at the hand's depth
    denom = view.depth - z  # +z is toward the camera
    u = f * x / denom
    v = f * y / denom
    xu = view.center[0] + u
    yi = view.center[1] - v * (640 / 480)  # y is in image-height units
    zm = -(z - z[0]) * view.scale * view.z_gain
    out = np.stack([1.0 - xu, yi, zm], axis=1)  # raw frame: mirror x
    if rng is not None and view.jitter > 0:
        sigma = view.jitter * 2.0 * view.scale
        noise = np.array(
            [
                [rng.gauss(0, sigma), rng.gauss(0, sigma), rng.gauss(0, sigma * 0.7)]
                for _ in range(21)
            ]
        )
        out = out + noise
    return out


def to_hand_frame(raw: np.ndarray, t_ns: int, hand: Hand, confidence: float = 0.95) -> HandFrame:
    return HandFrame(
        t_ns=t_ns,
        hand=hand,
        landmarks=tuple(Landmark(float(a), float(b), float(c)) for a, b, c in raw),
        confidence=confidence,
    )


POSE_FUNCS = {
    "open_palm": pose_open_palm,
    "fist": pose_fist,
    "h_left": pose_h,
    "h_right": pose_h,
    "point_up": pose_point_up,
    "two_up": pose_two_up,
    "pinch": pose_pinch,
    "none": pose_none,
}
SCALE_MAX = 0.5  # largest wrist->middle-MCP span as a fraction of the image width
KEEP_IN_FRAME = True
LEFT_HAND_P = 0.15  # share of left hands in static datasets (H signs are right-hand only in v0.1)
NONE_VIEW_MIX = 0.6  # share of 'none' samples rendered with another class's view regime
NONE_SIDEWAYS_P = 0.1  # share of V-sign 'none' poses rendered sideways like an H sign
STATIC_CLASSES = ("open_palm", "fist", "h_left", "h_right", "point_up", "two_up", "none")


def sample(
    cls: str,
    rng: random.Random,
    t_ns: int = 0,
    hand: Hand = Hand.RIGHT,
    preset: ViewPreset | None = None,
) -> HandFrame:
    """One labelled frame. H signs: thumb on top is the natural way to point sideways, so
    the back of the hand faces the camera when pointing left and the palm when pointing
    right (presets encode that; `flip_yaw_p` covers the other way round)."""
    pose = POSE_FUNCS[cls](rng)
    pose.hand = hand
    local = forward_kinematics(pose)
    for _attempt in range(40):
        if cls == "none" and preset is None and pose.sideways:
            view = random_view(rng, rng.choice(["h_left", "h_right"]))
        elif cls == "none" and preset is None and rng.random() < NONE_VIEW_MIX:
            view = random_view(
                rng, rng.choice(list(VIEW_PRESETS))
            )  # cover every class's view regime
        else:
            view = random_view(rng, cls, preset)
        if hand is Hand.LEFT:
            view.roll = -view.roll
        raw = render(local, view, rng)
        if cls not in ("h_left", "h_right"):
            break
        # the sign must read as pointing sideways on screen: wrist->index tip within 35 deg of horizontal
        ux, uy = (1.0 - raw[8, 0]) - (1.0 - raw[0, 0]), raw[8, 1] - raw[0, 1]
        ang = math.degrees(math.atan2(uy, ux))
        want = 180.0 if cls == "h_left" else 0.0
        if abs((ang - want + 180) % 360 - 180) <= 35:
            break
    # MediaPipe's handedness label is unreliable on sideways poses; mimic that.
    label = hand
    if cls in ("h_left", "h_right") and rng.random() < 0.5:
        label = Hand.LEFT if hand is Hand.RIGHT else Hand.RIGHT
    return to_hand_frame(raw, t_ns, label)


# --------------------------------------------------------------------------- sequences


def _lerp_pose(a: HandPose, b: HandPose, t: float) -> HandPose:
    p = HandPose(hand=a.hand)
    for f in FINGERS:
        fa, fb = a.fingers[f], b.fingers[f]
        p.fingers[f] = FingerPose(
            mcp=fa.mcp + (fb.mcp - fa.mcp) * t,
            pip=fa.pip + (fb.pip - fa.pip) * t,
            abd=fa.abd + (fb.abd - fa.abd) * t,
        )
    ta, tb = a.thumb, b.thumb
    p.thumb = ThumbPose(
        flex=ta.flex + (tb.flex - ta.flex) * t,
        abd=ta.abd + (tb.abd - ta.abd) * t,
        mcp=ta.mcp + (tb.mcp - ta.mcp) * t,
        ip=ta.ip + (tb.ip - ta.ip) * t,
    )
    return p


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def generate_sequence(
    kind: str, seconds: float, fps: float = 30.0, rng: random.Random | None = None, t0_ns: int = 0
) -> list[HandFrame]:
    """Frames for 'pinch_drag', 'palm_hold' or 'none_motion'. Timestamps are monotonic."""
    rng = rng or random.Random(0)
    n = max(1, int(seconds * fps))
    dt_ns = int(1e9 / fps)
    frames: list[HandFrame] = []
    base_view = random_view(
        rng, "pinch" if kind == "pinch_drag" else "open_palm" if kind == "palm_hold" else "none"
    )
    base_view.jitter = _u(rng, 0.003, 0.007)
    wobble = (_u(rng, -8, 8), _u(rng, -8, 8), _u(rng, -6, 6))
    if kind == "pinch_drag":
        # Leader first: a still open palm until the engine arms, then the pinch forms in
        # place, the pinched hand drags along a path, releases, and the hand is lost.
        # Facing the camera so the hand-written open_palm rule is confident, palm_scale 1
        # and modest z so the canonical 3-D thumb-index distance stays tight (< 0.3).
        view = View(
            yaw=_u(rng, -12, 12),
            pitch=_u(rng, 0, 18),
            roll=_u(rng, -8, 8),
            scale=_u(rng, 0.24, 0.34),
            center=(_u(rng, 0.35, 0.65), _u(rng, 0.4, 0.65)),
            depth=_u(rng, 12, 20),
            z_gain=_u(rng, 0.5, 0.7),
            jitter=_u(rng, 0.002, 0.004),
            palm_scale=1.0,
        )
        rest = pose_open_palm(rng)
        for f in FINGERS:  # keep the leader unambiguous: straight, spread fingers, thumb out
            rest.fingers[f].mcp = min(rest.fingers[f].mcp, 5.0)
            rest.fingers[f].pip = min(rest.fingers[f].pip, 8.0)
        rest.thumb = ThumbPose(
            flex=_u(rng, 0, 8), abd=_u(rng, 40, 55), mcp=_u(rng, 0, 8), ip=_u(rng, 0, 8)
        )
        # The leader must read as a confident open palm to the pipeline's own rule; resample
        # the view until it does (the rule is strict about thumb spread and foreshortening).
        from ..core.normalize import to_user_frame
        from ..core.recognizer import OPEN_PALM, RuleRecognizer

        rule = RuleRecognizer()
        for _try in range(60):
            probe = to_hand_frame(render(forward_kinematics(rest), view, None), 0, Hand.RIGHT)
            name, conf = rule.classify(to_user_frame(probe, True))
            if name == OPEN_PALM and conf >= 0.9:
                break
            view.yaw, view.pitch, view.roll = _u(rng, -12, 12), _u(rng, 0, 18), _u(rng, -8, 8)
            rest = pose_open_palm(rng)
            for f in FINGERS:
                rest.fingers[f].mcp = min(rest.fingers[f].mcp, 5.0)
                rest.fingers[f].pip = min(rest.fingers[f].pip, 8.0)
            rest.thumb = ThumbPose(
                flex=_u(rng, 0, 8), abd=_u(rng, 40, 60), mcp=_u(rng, 0, 8), ip=_u(rng, 0, 8)
            )
        pinch = pose_pinch(rng)
        for _retry in range(8):  # the drag must never release early: want a very tight pinch
            pts = forward_kinematics(pinch)
            if float(np.linalg.norm(pts[4] - pts[8])) < 0.12:
                break
            pinch = _solve_pinch(pose_pinch(rng), rng, iters=400)
        hold_s = _u(rng, 1.5, 1.8)
        form_s = _u(rng, 0.15, 0.3)
        release_s = 0.25
        drag_s = max(1.0, min(3.0, seconds - hold_s - form_s - release_s))
        t_form, t_drag, t_release = hold_s, hold_s + form_s, hold_s + form_s + drag_s
        n = int((t_release + release_s) * fps) + 1
        start = np.array(view.center)
        target = np.array([_u(rng, 0.15, 0.85), _u(rng, 0.2, 0.85)])
        ctrl = (start + target) / 2 + np.array([_u(rng, -0.2, 0.2), _u(rng, -0.2, 0.2)])
        for i in range(n):
            t = i / fps
            v = View(**view.__dict__)
            if t < t_form:
                pose, pos = rest, start  # still: only landmark jitter moves the wrist
            elif t < t_drag:
                pose, pos = _lerp_pose(rest, pinch, _smooth((t - t_form) / form_s)), start
            elif t < t_release:
                sdr = _smooth((t - t_drag) / drag_s)
                pos = (1 - sdr) ** 2 * start + 2 * (1 - sdr) * sdr * ctrl + sdr * sdr * target
                pose = pinch  # the solved pinch, unchanged, for the whole drag
                v.yaw += wobble[0] * 0.5 * math.sin(t * 1.3)
            else:
                sr = min(1.0, (t - t_release) / release_s)
                pose, pos = _lerp_pose(pinch, rest, _smooth(sr)), target
            v.center = (float(pos[0]), float(pos[1]))
            frames.append(
                to_hand_frame(
                    render(forward_kinematics(pose), v, rng), t0_ns + i * dt_ns, Hand.RIGHT
                )
            )
    elif kind == "palm_hold":
        pose = pose_open_palm(rng)
        for i in range(n):
            t = i / fps
            view = View(**base_view.__dict__)
            view.center = (
                base_view.center[0] + 0.004 * math.sin(t * 2.1),
                base_view.center[1] + 0.004 * math.cos(t * 1.7),
            )
            view.yaw += wobble[0] * 0.3 * math.sin(t)
            frames.append(
                to_hand_frame(
                    render(forward_kinematics(pose), view, rng), t0_ns + i * dt_ns, Hand.RIGHT
                )
            )
    elif kind == "none_motion":
        a, b = pose_none(rng), pose_none(rng)
        seg = _u(rng, 0.6, 1.5)
        for i in range(n):
            t = i / fps
            k = int(t / seg)
            if k > 0 and i > 0 and abs(t / seg - k) < 1e-9:
                a, b = b, pose_none(rng)
            pose = _lerp_pose(a, b, _smooth((t % seg) / seg))
            view = View(**base_view.__dict__)
            view.center = (
                base_view.center[0] + 0.15 * math.sin(t * 1.1),
                base_view.center[1] + 0.1 * math.sin(t * 0.7),
            )
            view.roll += 25 * math.sin(t * 0.8)
            frames.append(
                to_hand_frame(
                    render(forward_kinematics(pose), view, rng), t0_ns + i * dt_ns, Hand.RIGHT
                )
            )
    else:
        raise ValueError(kind)
    return frames


def write_frames(path: Path, frames: list[HandFrame], lost_after: bool = True) -> None:
    rec = Recorder(path)
    for hf in frames:
        rec.write(hf)
    if lost_after and frames:
        rec.write_lost(frames[-1].t_ns + 33_000_000)
    rec.close()


# --------------------------------------------------------------------------- CLI


def write_static(
    out: Path, per_class: int, seed: int, classes: tuple[str, ...] = STATIC_CLASSES + ("pinch",)
) -> None:
    rng = random.Random(seed)
    for cls in classes:
        rec = Recorder(out / cls / f"synth3d-{seed}.jsonl")
        for i in range(per_class):
            hand = (
                Hand.LEFT
                if rng.random() < LEFT_HAND_P and cls not in ("h_left", "h_right")
                else Hand.RIGHT
            )
            rec.write(sample(cls, rng, t_ns=i * 33_333_333, hand=hand))
        rec.close()
    # no_pinch: a mix of everything that is not a pinch
    rec = Recorder(out / "no_pinch" / f"synth3d-{seed}.jsonl")
    for i in range(per_class):
        cls = rng.choice(STATIC_CLASSES)
        rec.write(sample(cls, rng, t_ns=i * 33_333_333))
    rec.close()


def validate_against_real(real_dir: Path, per_class: int = 900, seed: int = 0) -> float:
    """Train on synthetic only, score on real recordings (files not starting with 'public-' or
    'synth'). Prints per-class recall and the confusion matrix; returns accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from ..core.normalize import features, to_user_frame
    from ..core.recorder import read_session

    rng = random.Random(seed)
    xs, ys = [], []
    for cls in STATIC_CLASSES:
        for _ in range(per_class):
            hand = (
                Hand.LEFT
                if rng.random() < LEFT_HAND_P and cls not in ("h_left", "h_right")
                else Hand.RIGHT
            )
            xs.append(features(to_user_frame(sample(cls, rng, hand=hand), True)))
            ys.append(cls)
    rx, ry = [], []
    for cls in STATIC_CLASSES:
        for f in sorted((real_dir / cls).glob("*.jsonl")):
            if f.name.startswith(("public-", "synth")):
                continue
            for r in read_session(f):
                if isinstance(r, HandFrame):
                    rx.append(features(to_user_frame(r, True)))
                    ry.append(cls)
    if not rx:
        print(f"no real recordings under {real_dir}")
        return 0.0
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(
        np.array(xs), np.array(ys)
    )
    pred = model.predict(np.array(rx))
    ry_a = np.array(ry)
    acc = float((pred == ry_a).mean())
    print(f"synthetic-only model on real data: accuracy={acc:.3f}")
    for cls in STATIC_CLASSES:
        sel = ry_a == cls
        print(f"  {cls:10s} recall={(pred[sel] == cls).mean():.2f} n={int(sel.sum())}")
    print("confusion rows=true cols=pred", list(STATIC_CLASSES))
    print(confusion_matrix(ry_a, pred, labels=list(STATIC_CLASSES)))
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="real datasets dir: train on synthetic, score on real, exit",
    )
    ap.add_argument("--out", type=Path, default=Path("datasets"))
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sequences",
        type=int,
        default=0,
        help="also write N pinch_drag sequences to tests/fixtures/synth3d",
    )
    ap.add_argument("--seq-out", type=Path, default=Path("tests/fixtures/synth3d"))
    args = ap.parse_args()
    if args.validate is not None:
        validate_against_real(args.validate, per_class=min(args.per_class, 1500), seed=args.seed)
        return 0
    write_static(args.out, args.per_class, args.seed)
    print(f"wrote {args.per_class} frames per class to {args.out}")
    if args.sequences:
        rng = random.Random(args.seed)
        for i in range(args.sequences):
            write_frames(
                args.seq_out / f"pinch_drag_{i}.jsonl",
                generate_sequence("pinch_drag", _u(rng, 2.0, 4.0), rng=rng),
            )
        write_frames(
            args.seq_out / "palm_hold_0.jsonl", generate_sequence("palm_hold", 3.0, rng=rng)
        )
        write_frames(
            args.seq_out / "none_motion_0.jsonl", generate_sequence("none_motion", 4.0, rng=rng)
        )
        print(
            f"wrote {args.sequences} pinch_drag sequences plus palm_hold and none_motion to {args.seq_out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
