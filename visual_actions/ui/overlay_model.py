"""View-model for the on-screen HUD: bus events in, one `Card` (or None) out per tick.

Pure Python, no AppKit, so the state → text / symbol / progress mapping is testable on
every platform. `ui/overlay.py` renders the card with PyObjC.

Precedence per render, highest first:
  ADJUST (and popup_ms after it)  the live system volume
  HOLDING / DRAGGING / REPEAT     persistent cards for states the user is in the middle of
  one-shot                        fired, failed, timed out, cancelled, dropped, missed, vetoed
                                  (shown for popup_ms, over the armed card if a menu is open)
  ARMED                           the open menu with its countdown
  nothing                         hidden
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.drag import DragEvent, DragPhase
from ..core.events import (
    ActionFired,
    HandLost,
    HandSeen,
    HoldProgress,
    ModeChanged,
    PalmVetoed,
    SnapPreview,
    TokenEmitted,
)
from ..core.modes import ADJUST, ARMED, DRAGGING, HOLDING, IDLE, REPEAT

# Accent names; the renderer maps them to system colours so they follow the appearance.
BLUE, GREEN, PURPLE, RED, ORANGE, GRAY = "blue", "green", "purple", "red", "orange", "gray"

NAMESPACE_SYMBOLS = {"window": "macwindow", "media": "music.note"}
NAMESPACE_TITLES = {"window": "Window", "media": "Media"}
LEADER_LABELS = {"window": "open palm", "media": "peace sign"}

# symbol name -> text glyph when the SF Symbol is missing on this macOS
FALLBACK_GLYPHS = {
    "hand.raised": "✋",
    "macwindow": "▭",
    "music.note": "♪",
    "checkmark.circle": "✓",
    "xmark.circle": "✕",
    "hand.point.up": "☝",
    "pause.circle": "⏸",
    "exclamationmark.triangle": "!",
    "speaker.wave.2": "♪",
    "speaker.slash": "♪",
    "hand.raised.slash": "✋",
    "arrow.left.and.right": "↔",
    "clock": "◷",
}


@dataclass(frozen=True)
class Card:
    state: str  # hold | armed | armed_lost | fired | failed | timeout | cancelled | drag | drag_lost | dropped | missed | repeat | adjust | veto
    symbol: str  # SF Symbol name
    title: str
    subtitle: str
    accent: str
    progress: float | None = None  # ring around the glyph, 0..1; None hides the ring
    bar_fraction: float | None = None  # thin bar along the bottom, 0..1; None hides it
    trailing: str = ""  # small label at the trailing edge (percentage, seconds left, repeat count)
    persistent: bool = False  # True: stays while the state lasts; False: auto-hides after popup_ms

    @property
    def glyph(self) -> str:
        return FALLBACK_GLYPHS.get(self.symbol, "•")


def _seconds(ns: int) -> str:
    return f"{max(0, ns) / 1e9:.1f} s"


class OverlayModel:
    """Feed it the bus events (all on one thread), call `render(now_ns)` on each tick."""

    def __init__(
        self,
        timeout_ns: int,
        popup_ns: int,
        grace_ns: int,
        repeat_window_ns: int,
    ) -> None:
        self.timeout_ns = timeout_ns
        self.popup_ns = popup_ns
        self.grace_ns = grace_ns
        self.repeat_window_ns = repeat_window_ns

        self.mode = IDLE
        self.namespace: str | None = None
        self.deadline_ns: int | None = None
        self.hold_fraction = 0.0
        self.hold_rate = 0.0
        self.hold_at_ns = 0
        self.last_token = ""
        self.hand_lost = False
        self.drag_window = ""
        self.drag_lost_since_ns: int | None = None
        self.snap_zone: str | None = None
        self.repeat_count = 0
        self.repeat_action = ""
        self.adjust_ready = False
        self.adjust_until_ns = 0
        self.oneshot: Card | None = None
        self.oneshot_until_ns = 0
        self.fired_at_ns: int | None = None

    # -- events -------------------------------------------------------------

    def on_mode(self, ev: ModeChanged) -> None:
        if ev.new != HOLDING or ev.namespace != self.namespace:
            self.hold_fraction, self.hold_rate = 0.0, 0.0  # a switch of root mode restarts the ring
        if ev.old != ARMED and ev.new == ARMED:
            self.last_token = ""  # the token that armed the menu is not a command
        if ev.new == REPEAT:
            self.repeat_count = 1 if ev.old != REPEAT else self.repeat_count + 1  # REPEAT -> REPEAT = fired again
        self.adjust_ready = ev.old == ADJUST and ev.new == ADJUST  # second ADJUST event = the pinch settled
        if ev.old == ADJUST and ev.new != ADJUST:
            self.adjust_until_ns = ev.t_ns + self.popup_ns
        if ev.new != DRAGGING:
            self.drag_lost_since_ns = None
            self.snap_zone = None
        if ev.new == IDLE and ev.old in (ARMED, REPEAT) and self.fired_at_ns != ev.t_ns:
            ns = NAMESPACE_TITLES.get(self.namespace or "", self.namespace or "")
            if ev.old == ARMED and self.deadline_ns is not None and ev.t_ns >= self.deadline_ns:
                self._flash(Card("timeout", "clock", "Timed out", f"{ns} menu closed", GRAY), ev.t_ns)
            else:
                self._flash(Card("cancelled", "xmark.circle", "Cancelled", f"{ns} menu closed", GRAY), ev.t_ns)
        self.mode, self.deadline_ns, self.namespace = ev.new, ev.deadline_ns, ev.namespace

    def on_hold(self, ev: HoldProgress) -> None:
        self.hold_fraction, self.hold_rate, self.hold_at_ns = ev.fraction, ev.rate, ev.t_ns

    def on_token(self, ev: TokenEmitted) -> None:
        if ev.token.name != "none":
            self.last_token = ev.token.name

    def on_action(self, ev: ActionFired) -> None:
        self.fired_at_ns = ev.t_ns
        self.repeat_action = ev.action.name
        if self.mode == ADJUST and ev.ok:
            return  # volume steps: the level on the card is the feedback
        if ev.ok:
            self._flash(Card("fired", "checkmark.circle", ev.action.name, "fired", GREEN), ev.t_ns)
        else:
            self._flash(Card("failed", "exclamationmark.triangle", ev.action.name, ev.message or "failed", RED), ev.t_ns)

    def on_drag(self, ev: DragEvent) -> None:
        if ev.phase is DragPhase.PAUSE:
            self.drag_lost_since_ns = ev.t_ns
        elif ev.phase is DragPhase.RESUME:
            self.drag_lost_since_ns = None
        elif ev.phase is DragPhase.END:
            self.drag_lost_since_ns = None
            title = f"Snapped {ev.snapped.replace('_', ' ')}" if ev.snapped else "Dropped"
            self._flash(Card("dropped", "checkmark.circle", title, ev.window, PURPLE), ev.t_ns)
        elif ev.phase is DragPhase.MISS:
            self._flash(Card("missed", "xmark.circle", "Nothing to grab", "pinch over a window", GRAY), ev.t_ns)
        if ev.phase in (DragPhase.START, DragPhase.MOVE, DragPhase.RESUME, DragPhase.PAUSE):
            self.drag_window = ev.window
        else:
            self.drag_window = ""

    def on_snap(self, ev: SnapPreview) -> None:
        self.snap_zone = ev.zone

    def on_hand_lost(self, ev: HandLost) -> None:
        self.hand_lost = True

    def on_hand_seen(self, ev: HandSeen) -> None:
        self.hand_lost = False

    def on_veto(self, ev: PalmVetoed) -> None:
        if self.mode == IDLE:
            self._flash(Card("veto", "hand.raised.slash", "Palm on face", "ignored", ORANGE), ev.t_ns)

    def _flash(self, card: Card, t_ns: int) -> None:
        self.oneshot, self.oneshot_until_ns = card, t_ns + self.popup_ns

    # -- render -------------------------------------------------------------

    def hold_progress(self, now: int) -> float:
        """Evidence fraction extrapolated at the token's rate, so the ring is smooth between tokens."""
        return min(1.0, self.hold_fraction + max(0.0, self.hold_rate) * (now - self.hold_at_ns) / 1e9)

    def render(self, now: int, volume: tuple[float, bool] | None = None) -> Card | None:
        ns = self.namespace or "window"
        if self.mode == ADJUST or (self.mode == IDLE and now < self.adjust_until_ns):
            return self._adjust_card(volume)
        if self.mode == HOLDING:
            p = self.hold_progress(now)
            return Card(
                "hold",
                "hand.raised",
                f"Hold the {LEADER_LABELS.get(ns, ns)}",
                f"{NAMESPACE_TITLES.get(ns, ns)} menu",
                BLUE,
                progress=p,
                trailing=f"{min(100, int(p * 100))}%",
                persistent=True,
            )
        if self.mode == DRAGGING:
            if self.drag_lost_since_ns is not None:
                left = max(0, self.grace_ns - (now - self.drag_lost_since_ns))
                return Card(
                    "drag_lost",
                    "pause.circle",
                    "Hand lost",
                    f"{self.drag_window} waits for the pinch",
                    RED,
                    bar_fraction=left / self.grace_ns if self.grace_ns else 0.0,
                    trailing=_seconds(left),
                    persistent=True,
                )
            zone = (self.snap_zone or "").replace("_", " ")
            return Card("drag", "hand.point.up", "Dragging", self.drag_window, PURPLE, trailing=zone, persistent=True)
        if self.mode == REPEAT:
            left = max(0, (self.deadline_ns or now) - now)
            return Card(
                "repeat",
                "arrow.left.and.right",
                self.repeat_action or self.last_token,
                "slide sideways to repeat",
                ORANGE,
                bar_fraction=left / self.repeat_window_ns if self.repeat_window_ns else 0.0,
                trailing=f"×{self.repeat_count}",
                persistent=True,
            )
        if self.oneshot is not None and now < self.oneshot_until_ns:
            return self.oneshot
        if self.mode == ARMED:
            left = max(0, (self.deadline_ns or now) - now)
            fraction = left / self.timeout_ns if self.timeout_ns else 0.0
            title = f"{NAMESPACE_TITLES.get(ns, ns)} menu"
            if self.hand_lost:
                return Card(
                    "armed_lost",
                    "pause.circle",
                    title,
                    "hand lost · the menu stays open",
                    RED,
                    bar_fraction=fraction,
                    trailing=_seconds(left),
                    persistent=True,
                )
            return Card(
                "armed",
                NAMESPACE_SYMBOLS.get(ns, "macwindow"),
                title,
                f"last: {self.last_token}" if self.last_token else "show a gesture",
                GREEN,
                bar_fraction=fraction,
                trailing=_seconds(left),
                persistent=True,
            )
        return None

    def _adjust_card(self, volume: tuple[float, bool] | None) -> Card:
        if volume is None:
            title, level, muted = "Volume", 1.0, False
        else:
            level, muted = volume
            title = f"Volume {round(level * 100)}%" + (" (muted)" if muted else "")
        if self.mode != ADJUST:
            sub = "done"
        elif self.adjust_ready:
            sub = "◀ quieter · pinch · louder ▶"
        else:
            sub = "hold the pinch still…"
        return Card(
            "adjust",
            "speaker.slash" if muted else "speaker.wave.2",
            title,
            sub,
            PURPLE,
            progress=level,
            trailing="muted" if muted else "",
            persistent=self.mode == ADJUST,
        )
