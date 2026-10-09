"""Always-on-top indicator: mode label, countdown ring, last token, action flash.

Main thread only. Subscribes to the bus and redraws on Tick.
"""

from __future__ import annotations

from collections.abc import Callable

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSString

from ..core.drag import DragEvent, DragPhase
from ..core.events import ActionFired, Bus, HoldProgress, ModeChanged, Tick, TokenEmitted
from ..core.modes import ADJUST, ARMED, DRAGGING, HOLDING, IDLE, REPEAT

W, H = 260, 72
LEADER_LABELS = {"window": "palm", "media": "peace"}


class OverlayView(NSView):
    state = objc.ivar()

    def initWithFrame_(self, frame):
        self = objc.super(OverlayView, self).initWithFrame_(frame)  # noqa: PLW0642 - PyObjC idiom
        if self is None:
            return None
        self.state = {"mode": IDLE, "progress": 0.0, "label": "", "sub": "", "flash": False}
        return self

    def drawRect_(self, rect):
        s = self.state
        bg = NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.86)
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(self.bounds(), 14, 14)
        bg.setFill()
        path.fill()

        cx, cy, r = 36, H / 2, 22
        track = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.18)
        ring = NSBezierPath.bezierPath()
        ring.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((cx, cy), r, 0, 360, False)
        ring.setLineWidth_(5)
        track.setStroke()
        ring.stroke()
        if s["progress"] > 0:
            color = {
                HOLDING: NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.65, 1.0, 1.0),
                ARMED: NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.85, 0.45, 1.0),
                DRAGGING: NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.55, 1.0, 1.0),
                "lost": NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.3, 0.35, 1.0),
                REPEAT: NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.75, 0.3, 1.0),
                ADJUST: NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.55, 1.0, 1.0),
            }.get(s["mode"], NSColor.whiteColor())
            if s["flash"]:
                color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.8, 0.2, 1.0)
            arc = NSBezierPath.bezierPath()
            end = 90 - 360 * min(1.0, s["progress"])
            arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((cx, cy), r, 90, end, True)
            arc.setLineWidth_(5)
            arc.setLineCapStyle_(1)
            color.setStroke()
            arc.stroke()

        attrs = {NSFontAttributeName: NSFont.boldSystemFontOfSize_(16), NSForegroundColorAttributeName: NSColor.whiteColor()}
        NSString.stringWithString_(s["label"]).drawAtPoint_withAttributes_((72, cy + 2), attrs)
        sub_attrs = {
            NSFontAttributeName: NSFont.systemFontOfSize_(12),
            NSForegroundColorAttributeName: NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.7),
        }
        NSString.stringWithString_(s["sub"]).drawAtPoint_withAttributes_((72, cy - 18), sub_attrs)


class Overlay:
    def __init__(
        self,
        bus: Bus,
        hold_ns: int,
        timeout_ns: int,
        popup_ms: int,
        grace_ns: int = 2_500_000_000,
        volume: Callable[[], tuple[float, bool] | None] | None = None,
    ) -> None:
        self.volume = volume  # reads the real system volume; polled only while adjusting
        self._volume_cache: tuple[float, bool] | None = None
        self._volume_read_ns = 0
        self.adjust_until_ns = 0  # keep showing the volume for popup_ms after the pinch is released
        self.hold_ns, self.timeout_ns, self.popup_ns = hold_ns, timeout_ns, popup_ms * 1_000_000
        self.grace_ns = grace_ns
        self.lost_since_ns: int | None = None
        self.mode = IDLE
        self.adjust_ready = False
        self.namespace: str | None = None
        self.mode_since_ns = 0
        self.hold_fraction = 0.0
        self.hold_rate = 0.0
        self.hold_at_ns = 0
        self.deadline_ns: int | None = None
        self.last_token = ""
        self.drag_window = ""
        self.flash_until_ns = 0
        self.flash_text = ""

        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + (screen.size.width - W) / 2
        y = screen.origin.y + screen.size.height - H - 12
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, W, H),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self.panel.setLevel_(NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHasShadow_(True)
        self.panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)
        self.view = OverlayView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        self.panel.setContentView_(self.view)

        bus.subscribe(ModeChanged, self._on_mode)
        bus.subscribe(HoldProgress, self._on_hold)
        bus.subscribe(TokenEmitted, self._on_token)
        bus.subscribe(DragEvent, self._on_drag)
        bus.subscribe(ActionFired, self._on_action)
        bus.subscribe(Tick, self._on_tick)

    def close(self) -> None:
        """Hide the panel; the app drops the bus this overlay listens on when it rebuilds from a new config."""
        if self.panel.isVisible():
            self.panel.orderOut_(None)

    def _on_drag(self, ev: DragEvent) -> None:
        if ev.phase is DragPhase.PAUSE:
            self.lost_since_ns = ev.t_ns
        elif ev.phase in (DragPhase.RESUME, DragPhase.END):
            self.lost_since_ns = None
        self.drag_window = ev.window if ev.phase in (DragPhase.START, DragPhase.MOVE, DragPhase.RESUME, DragPhase.PAUSE) else ""

    def _on_mode(self, ev: ModeChanged) -> None:
        if ev.new != HOLDING or ev.namespace != self.namespace:
            self.hold_fraction, self.hold_rate = 0.0, 0.0  # a switch of root mode restarts the ring
        self.mode, self.mode_since_ns, self.deadline_ns = ev.new, ev.t_ns, ev.deadline_ns
        self.namespace = ev.namespace
        self.adjust_ready = ev.old == ADJUST and ev.new == ADJUST  # second ADJUST event = the pinch settled
        if ev.old == ADJUST and ev.new != ADJUST:
            self.adjust_until_ns = ev.t_ns + self.popup_ns
        if ev.new != DRAGGING:
            self.lost_since_ns = None

    def _on_hold(self, ev: HoldProgress) -> None:
        self.hold_fraction, self.hold_rate, self.hold_at_ns = ev.fraction, ev.rate, ev.t_ns

    def _on_token(self, ev: TokenEmitted) -> None:
        self.last_token = ev.token.name

    def _on_action(self, ev: ActionFired) -> None:
        self.flash_text = ev.action.name if ev.ok else f"{ev.action.name} failed"
        self.flash_until_ns = ev.t_ns + self.popup_ns

    def _read_volume(self, now: int) -> tuple[float, bool] | None:
        if self.volume is None:
            return None
        if now - self._volume_read_ns >= 50_000_000:  # ~1 ms CoreAudio read, 20 times a second at most
            self._volume_read_ns = now
            try:
                self._volume_cache = self.volume()
            except Exception:  # noqa: BLE001 - a failed read shows "volume" without a number
                self._volume_cache = None
        return self._volume_cache

    def _on_tick(self, ev: Tick) -> None:
        now = ev.t_ns
        s = self.view.state
        flashing = now < self.flash_until_ns
        if self.mode == ADJUST or (self.mode == IDLE and now < self.adjust_until_ns):
            # the live system volume, read back from the OS, not a count of our own key presses
            vol = self._read_volume(now)
            if vol is None:
                label, level = "volume", 1.0
            else:
                level = vol[0]
                label = f"Volume {round(level * 100)}%" + (" (muted)" if vol[1] else "")
            if self.mode != ADJUST:
                sub = "done"
            elif self.adjust_ready:
                sub = "◀ quieter · pinch · louder ▶"
            else:
                sub = "hold the pinch still…"
            s.update(mode=ADJUST, progress=level, label=label, sub=sub, flash=False)
        elif flashing:
            s.update(mode=self.mode, progress=1.0, label=self.flash_text, sub="fired", flash=True)
        elif self.mode == HOLDING:
            p = min(1.0, self.hold_fraction + max(0.0, self.hold_rate) * (now - self.hold_at_ns) / 1e9)
            shape = LEADER_LABELS.get(self.namespace or "", self.namespace or "palm")
            s.update(mode=HOLDING, progress=p, label=f"Hold… {self.namespace or ''}", sub=f"{shape} {min(100, int(p * 100))}%", flash=False)
        elif self.mode == DRAGGING and self.lost_since_ns is not None:
            left = max(0, self.grace_ns - (now - self.lost_since_ns))
            s.update(mode="lost", progress=left / self.grace_ns, label="hand lost", sub=f"{left / 1e9:.1f}s to resume", flash=False)
        elif self.mode == DRAGGING:
            s.update(mode=DRAGGING, progress=1.0, label="drag", sub=self.drag_window[:34], flash=False)
        elif self.mode == REPEAT and self.deadline_ns:
            left = max(0, self.deadline_ns - now)
            s.update(mode=REPEAT, progress=left / 1.5e9, label="slide to repeat", sub=f"{self.flash_text or self.last_token}", flash=False)
        elif self.mode == ARMED and self.deadline_ns:
            left = max(0, self.deadline_ns - now)
            s.update(mode=ARMED, progress=left / self.timeout_ns, label=self.namespace or "window", sub=f"{left / 1e9:.1f}s · {self.last_token}", flash=False)
        else:
            if self.panel.isVisible():
                self.panel.orderOut_(None)
            return
        if not self.panel.isVisible():
            self.panel.orderFrontRegardless()
        self.view.setNeedsDisplay_(True)
