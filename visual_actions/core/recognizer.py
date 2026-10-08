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

FINGERS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)


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
    def __init__(
        self,
        extended: float = 0.7,
        curled: float = 0.45,
        thumb_out: float = 0.35,
        direction_deg: float = 35.0,
    ) -> None:
        self.extended = extended
        self.curled = curled
        self.thumb_out = thumb_out
        self.direction_cos = math.cos(math.radians(direction_deg))

    def classify(self, hf: HandFrame) -> tuple[str, float]:
        c = canonical(hf)
        ext = [_extension(c, *f) for f in FINGERS]
        index, middle, ring, pinky = ext
        thumb_idx = float(np.linalg.norm(c.pts[THUMB_TIP][:2] - c.pts[INDEX_MCP][:2]))
        all_ext = all(e >= self.extended for e in ext)
        all_curl = all(e <= self.curled for e in ext)
        if all_ext and thumb_idx >= self.thumb_out:
            return OPEN_PALM, min(ext)
        if all_curl:
            return FIST, 1.0 - max(ext)
        v = c.pts[INDEX_TIP][:2] - c.pts[INDEX_MCP][:2]
        n = float(np.linalg.norm(v))
        if n > 1e-6 and index >= self.extended and ring <= self.curled and pinky <= self.curled:
            cos_left, cos_up = -v[0] / n, -v[1] / n
            if middle >= self.extended:
                conf = min(index, middle, 1.0 - ring, 1.0 - pinky)
                if cos_left >= self.direction_cos:
                    return H_LEFT, conf
                if -cos_left >= self.direction_cos:
                    return H_RIGHT, conf
                if cos_up >= self.direction_cos:
                    return TWO_UP, conf
            elif middle <= self.curled and cos_up >= self.direction_cos:
                return POINT_UP, min(index, 1.0 - middle, 1.0 - ring, 1.0 - pinky)
        return NONE, 0.5


class SklearnRecognizer:
    """A scikit-learn classifier on the shared feature vector. Classes include 'none'."""

    def __init__(self, model_path: Path) -> None:
        import joblib

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
