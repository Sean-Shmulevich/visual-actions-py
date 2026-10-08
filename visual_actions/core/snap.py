"""Edge snapping for drags, platform-independent.

Why our own: BetterTouchTool, Rectangle, macOS tiling and Windows Snap all key off
real mouse drags of a title bar. We move windows through accessibility APIs, which
none of them see, so snapping has to live here. It is pure geometry over the
display's visible frame and runs headless.

UX rules encoded:
- Zones arm when the POINTER (not the window) nears an edge: halves at the left/right
  edges, maximize at the top edge, quarters near corners. Corners win over edges.
- A zone previews only after a short dwell, so sweeping across the screen never
  flashes previews.
- Leaving a zone needs more distance than entering it (hysteresis), so a hand that
  trembles on the threshold does not flicker.
- Snapping happens on release, never mid-drag; the drag itself is unaffected.
- A snapped window that is dragged again is restored to its pre-snap size first, so
  snapping is reversible (see DragController).
"""

from __future__ import annotations

from dataclasses import dataclass

from .automation import Rect


@dataclass(frozen=True)
class SnapZone:
    name: str  # left, right, top, top_left, top_right, bottom_left, bottom_right
    target: Rect  # window frame on release


@dataclass(frozen=True)
class SnapRules:
    edge_px: float = 28.0
    corner_px: float = 110.0
    dwell_ns: int = 150_000_000
    quarters: bool = True
    maximize: bool = True
    leave_factor: float = 2.0  # hysteresis: leave distance = enter distance * factor


class SnapEngine:
    def __init__(self, visible: Rect, rules: SnapRules = SnapRules()) -> None:
        self.visible = visible
        self.rules = rules
        self._candidate: SnapZone | None = None
        self._candidate_since_ns: int | None = None
        self.active: SnapZone | None = None

    # -- geometry -------------------------------------------------------------

    def zone_at(self, x: float, y: float, armed: SnapZone | None = None) -> SnapZone | None:
        """The zone the pointer is in, with hysteresis relative to `armed`."""
        v = self.visible
        r = self.rules
        f = r.leave_factor if armed is not None else 1.0
        near_left = x - v.x <= r.edge_px * f
        near_right = (v.x + v.w) - x <= r.edge_px * f
        near_top = y - v.y <= r.edge_px * f
        near_bottom = (v.y + v.h) - y <= r.edge_px * f
        cx = r.corner_px * f
        corner_left = x - v.x <= cx
        corner_right = (v.x + v.w) - x <= cx
        corner_top = y - v.y <= cx
        corner_bottom = (v.y + v.h) - y <= cx
        hw, hh = v.w // 2, v.h // 2
        if r.quarters:
            if corner_left and corner_top:
                return SnapZone("top_left", Rect(v.x, v.y, hw, hh))
            if corner_right and corner_top:
                return SnapZone("top_right", Rect(v.x + hw, v.y, v.w - hw, hh))
            if corner_left and corner_bottom:
                return SnapZone("bottom_left", Rect(v.x, v.y + hh, hw, v.h - hh))
            if corner_right and corner_bottom:
                return SnapZone("bottom_right", Rect(v.x + hw, v.y + hh, v.w - hw, v.h - hh))
        if near_left:
            return SnapZone("left", Rect(v.x, v.y, hw, v.h))
        if near_right:
            return SnapZone("right", Rect(v.x + hw, v.y, v.w - hw, v.h))
        if r.maximize and near_top:
            return SnapZone("top", Rect(v.x, v.y, v.w, v.h))
        if armed is not None and near_bottom and armed.name in ("left", "right", "top"):
            return armed  # bottom edge keeps whatever is armed; it has no zone of its own
        return None

    # -- state ----------------------------------------------------------------

    def update(self, x: float, y: float, t_ns: int) -> SnapZone | None:
        """Feed the pointer; returns the active (previewed) zone after dwell + hysteresis."""
        if self.active is not None:
            stay = self.zone_at(x, y, armed=self.active)
            if stay is not None and stay.name == self.active.name:
                return self.active
            # moved out of the active zone (beyond the hysteresis band) or into another
            self.active = None
        z = self.zone_at(x, y)
        if z is None:
            self._candidate, self._candidate_since_ns = None, None
            return None
        if self._candidate is None or self._candidate.name != z.name:
            self._candidate, self._candidate_since_ns = z, t_ns
            return None
        assert self._candidate_since_ns is not None
        if t_ns - self._candidate_since_ns >= self.rules.dwell_ns:
            self.active = z
            self._candidate, self._candidate_since_ns = None, None
        return self.active

    def reset(self) -> None:
        self._candidate, self._candidate_since_ns, self.active = None, None, None
