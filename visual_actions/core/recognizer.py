"""Tier 2: features -> (gesture name, confidence), plus the smoother that emits tokens."""

from __future__ import annotations

import math
from collections import Counter, deque
from pathlib import Path
from typing import Protocol

import numpy as np

from .normalize import Canonical, canonical, features
from .types import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandFrame,
    Token,
)

NONE = "none"
OPEN_PALM = "open_palm"
FIST = "fist"
H_LEFT = "h_left"
H_RIGHT = "h_right"
POINT_UP = "point_up"
TWO_UP = "two_up"
MIDDLE_UP = "middle_up"  # you know the one
THUMBS_UP = "thumbs_up"
THUMBS_DOWN = "thumbs_down"
PALM_SIDE = "palm_side"  # right hand edge-on to the camera, fingers together pointing sideways, thumb on top: the scroll hand

FINGERS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)


def tip_spread(hf: HandFrame) -> float:
    """Index-tip to pinky-tip distance in canonical hand units (~0.6-1.0 for a spread palm)."""
    c = canonical(hf)
    return float(np.linalg.norm(c.pts[INDEX_TIP][:2] - c.pts[PINKY_TIP][:2]))


class Recognizer(Protocol):
    def classify(self, hf: HandFrame) -> tuple[str, float]: ...


def _extension(c: Canonical, mcp: int, pip: int, tip: int) -> float:
    """0 = tip on the MCP (curled), ~1 = fully straight."""
    full = np.linalg.norm(c.pts[pip] - c.pts[mcp]) + np.linalg.norm(c.pts[tip] - c.pts[pip])
    if full < 1e-6:
        return 0.0
    return float(np.linalg.norm(c.pts[tip] - c.pts[mcp]) / full)


class RuleRecognizer:
    """Hand-written rules. Drives the leader and escape, and covers the H sign until a
    model is trained. Thresholds are on canonical coordinates (wrist->middle MCP = 1)."""

    # Thresholds from 2026-10-08 webcam samples (tests/fixtures/real): straight fingers
    # score ~0.8-0.95 on the path-ratio metric, curled ring/pinky in the H sign ~0.2-0.35,
    # thumb-tip-to-index-MCP ~0.55 open vs ~0.16 tucked. A fist seen knuckles-first is
    # ambiguous in 2D; the trained tier 2 model is the real answer for that.
    # The peace sign (TWO_UP) is looser: the thumb pins ring and pinky only half down, so
    # they score ~0.5 (p90 0.6) on the user's two_up recording. At peace_curled = 0.6 the
    # rule reads 88 % of those frames versus 33 % at 0.45, and no other recorded class gains
    # a confident still two_up token. Its confidence is the straight-vs-folded separation.
    # Thumbs up/down are checked before the fist (HaGRID "like"/"dislike" read as fist
    # about half the time, and a fist cancels): a straight thumb whose tip is clear of
    # every other joint above (or below) by thumb_clear, pointing within ~53 deg of
    # vertical, with every finger tip folded back within thumb_fold of its knuckle.
    # The wrist must be in frame: a hand half out through the bottom edge has its
    # fingers guessed as folded and read as thumbs up in the user's sessions.
    def __init__(
        self,
        extended: float = 0.7,
        curled: float = 0.45,
        thumb_out: float = 0.35,
        direction_deg: float = 35.0,
        peace_curled: float = 0.6,
        peace_separation: float = 0.5,
        thumb_clear: float = 0.25,
        thumb_fold: float = 0.6,
        thumb_cos: float = 0.6,
        thumb_wrist_max_y: float = 0.92,
        side_max_wrist_y: float = 0.85,
    ) -> None:
        self.side_max_wrist_y = side_max_wrist_y
        self.thumb_wrist_max_y = thumb_wrist_max_y
        self.thumb_clear = thumb_clear
        self.thumb_fold = thumb_fold
        self.thumb_cos = thumb_cos
        self.extended = extended
        self.curled = curled
        self.peace_curled = peace_curled
        self.peace_separation = peace_separation
        self.thumb_out = thumb_out
        self.direction_cos = math.cos(math.radians(direction_deg))

    def classify(self, hf: HandFrame) -> tuple[str, float]:
        c = canonical(hf)
        ext = [_extension(c, *f) for f in FINGERS]
        index, middle, ring, pinky = ext
        thumb_idx = float(np.linalg.norm(c.pts[THUMB_TIP][:2] - c.pts[INDEX_MCP][:2]))
        all_ext = all(e >= self.extended for e in ext)
        all_curl = all(e <= self.curled for e in ext)
        if all_ext:
            # The scroll hand: all four fingers extended and pointing sideways (arm level). The
            # leader palm points up (2026-10-09 session: |cos| 0.03 over a whole hold) while the
            # flat sideways hand reads 0.95..1.0 whatever its tilt; knuckle width and tip spread
            # swing between 0.1 and 0.45 as the palm tilts, so they are not used. Checked before
            # the open palm so the scroll hand can never arm the menu, and the face veto (open
            # palm only) never fires on it.
            # The arm is raised (wrist in the top 85 % of the frame): hands resting flat on the desk
            # also point sideways but sit at the bottom edge (p50 wrist y 0.90..0.97 in the sessions
            # versus 0.67 for the scroll hand).
            vi = c.pts[INDEX_TIP][:2] - c.pts[INDEX_MCP][:2]
            ni = float(np.linalg.norm(vi))
            sideways = abs(vi[0]) / ni if ni > 1e-6 else 0.0
            if sideways >= self.direction_cos and hf.landmarks[WRIST].y <= self.side_max_wrist_y:
                return PALM_SIDE, min(min(ext), (sideways - self.direction_cos) / (1.0 - self.direction_cos))
        if all_ext and thumb_idx >= self.thumb_out:
            return OPEN_PALM, min(ext)
        thumbs = self._thumbs(c) if hf.landmarks[WRIST].y <= self.thumb_wrist_max_y else None
        if thumbs is not None:
            return thumbs
        if all_curl:
            return FIST, 1.0 - max(ext)
        if middle >= self.extended and index <= self.curled and ring <= self.curled and pinky <= self.curled:
            vm = c.pts[MIDDLE_TIP][:2] - c.pts[MIDDLE_MCP][:2]
            nm = float(np.linalg.norm(vm))
            if nm > 1e-6 and -vm[1] / nm >= self.direction_cos:
                return MIDDLE_UP, min(middle, 1.0 - index, 1.0 - ring, 1.0 - pinky)
        v = c.pts[INDEX_TIP][:2] - c.pts[INDEX_MCP][:2]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return NONE, 0.5
        cos_left, cos_up = -v[0] / n, -v[1] / n
        if (
            index >= self.extended
            and middle >= self.extended
            and ring <= self.peace_curled
            and pinky <= self.peace_curled
            and cos_up >= self.direction_cos
        ):
            straight, folded = min(index, middle), max(ring, pinky)
            return TWO_UP, min(straight, max(0.0, min(1.0, (straight - folded) / self.peace_separation)))
        if index >= self.extended and ring <= self.curled and pinky <= self.curled:
            if middle >= self.extended:
                conf = min(index, middle, 1.0 - ring, 1.0 - pinky)
                if cos_left >= self.direction_cos:
                    return H_LEFT, conf
                if -cos_left >= self.direction_cos:
                    return H_RIGHT, conf
            elif middle <= self.curled and cos_up >= self.direction_cos:
                return POINT_UP, min(index, 1.0 - middle, 1.0 - ring, 1.0 - pinky)
        return NONE, 0.5


    def _thumbs(self, c: Canonical) -> tuple[str, float] | None:
        p = c.pts[:, :2]
        fold = max(float(np.linalg.norm(p[tip] - p[mcp])) for mcp, _, tip in FINGERS)
        if fold > self.thumb_fold:
            return None
        v = p[THUMB_TIP] - p[THUMB_MCP]
        n = float(np.linalg.norm(v))
        full = float(np.linalg.norm(p[THUMB_IP] - p[THUMB_MCP]) + np.linalg.norm(p[THUMB_TIP] - p[THUMB_IP]))
        if n < 1e-6 or n / max(full, 1e-6) < self.extended:
            return None  # a bent thumb is a fist
        others = [j for f in FINGERS for j in f]
        ys = p[others, 1]
        cos_up = -v[1] / n
        above = float(ys.min() - p[THUMB_TIP][1])
        below = float(p[THUMB_TIP][1] - ys.max())
        fold_conf = min(1.0, (self.thumb_fold + 0.15 - fold) / 0.35)
        if cos_up >= self.thumb_cos and above >= self.thumb_clear:
            return THUMBS_UP, min(fold_conf, min(1.0, above / (2 * self.thumb_clear)))
        if -cos_up >= self.thumb_cos and below >= self.thumb_clear:
            return THUMBS_DOWN, min(fold_conf, min(1.0, below / (2 * self.thumb_clear)))
        return None


class SklearnRecognizer:
    """A scikit-learn classifier on the shared feature vector. Classes include 'none'."""

    def __init__(self, model_path: Path) -> None:
        import joblib

        self.path = model_path
        self._model = joblib.load(model_path)
        self._classes = list(self._model.classes_)

    def classify(self, hf: HandFrame) -> tuple[str, float]:
        probs = self._model.predict_proba(features(hf).reshape(1, -1))[0]
        i = int(np.argmax(probs))
        return str(self._classes[i]), float(probs[i])


class CompositeRecognizer:
    """Rules first; a confident rule wins, otherwise the model (if any) answers."""

    def __init__(self, rules: Recognizer, model: Recognizer | None, rule_min: float = 0.9) -> None:
        self.rules = rules
        self.model = model
        self.rule_min = rule_min

    @property
    def model_path(self) -> Path | None:
        """The joblib the model tier loaded, or None when rules run alone (for a session's meta.json)."""
        return getattr(self.model, "path", None)

    def classify(self, hf: HandFrame) -> tuple[str, float]:
        name, conf = self.rules.classify(hf)
        if self.model is None or (name != NONE and conf >= self.rule_min):
            return name, conf
        return self.model.classify(hf)


class Smoother:
    """Majority vote over a window with a stillness check. Emits at most one Token per
    window length, including 'none' tokens so the mode engine can see an idle hand."""

    def __init__(self, window_ns: int = 250_000_000, still_px: float = 12.0, frame_width_px: int = 640) -> None:
        self.window_ns = window_ns
        self.still_px = still_px
        self.frame_width_px = frame_width_px
        self._buf: deque[tuple[int, str, float, float, float]] = deque()
        self._last_emit_ns: int | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._last_emit_ns = None

    def push(self, hf: HandFrame, name: str, conf: float) -> Token | None:
        w = hf.landmarks[WRIST]
        self._buf.append((hf.t_ns, name, conf, w.x * self.frame_width_px, w.y * self.frame_width_px))
        while self._buf and hf.t_ns - self._buf[0][0] > self.window_ns:
            self._buf.popleft()
        if self._last_emit_ns is not None and hf.t_ns - self._last_emit_ns < self.window_ns:
            return None
        if self._buf[-1][0] - self._buf[0][0] < self.window_ns * 0.8:
            return None  # window not full yet
        votes = Counter(n for _, n, _, _, _ in self._buf)
        name, count = votes.most_common(1)[0]
        confs = [c for _, n, c, _, _ in self._buf if n == name]
        xs = [x for _, _, _, x, _ in self._buf]
        ys = [y for _, _, _, _, y in self._buf]
        drift = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        self._last_emit_ns = hf.t_ns
        return Token(
            t_ns=hf.t_ns,
            name=name,
            confidence=(sum(confs) / len(confs)) * (count / len(self._buf)),
            hand=hf.hand,
            still=drift <= self.still_px,
            x=w.x,
            y=w.y,
        )
