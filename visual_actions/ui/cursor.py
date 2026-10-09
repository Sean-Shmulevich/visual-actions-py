"""Two always-on-top layers, main thread only:

- a small pinch cursor: hollow ring while armed (where a pinch would land), filled dot
  while dragging;
- a translucent snap preview rectangle over the frame the window would snap to.
"""

from __future__ import annotations

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
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

from ..core.drag import DragEvent, DragPhase
from ..core.events import Bus, ModeChanged, PointerMoved, SnapPreview, Tick
from ..core.modes import ARMED, DRAGGING, SCROLL
from ..core.scroll import ScrollEvent, ScrollPhase

CURSOR_SIZE = 26


def _panel(x: float, y: float, w: float, h: float, level_offset: int = 0) -> NSPanel:
    p = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, w, h),
        NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
        NSBackingStoreBuffered,
        False,
    )
    p.setLevel_(NSStatusWindowLevel + level_offset)
    p.setOpaque_(False)
    p.setBackgroundColor_(NSColor.clearColor())
    p.setIgnoresMouseEvents_(True)
    p.setHasShadow_(False)
    p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)
    return p


class CursorView(NSView):
    filled = objc.ivar()

    def initWithFrame_(self, frame):
        self = objc.super(CursorView, self).initWithFrame_(frame)  # noqa: PLW0642 - PyObjC idiom
        if self is None:
            return None
        self.filled = False
        return self

    def drawRect_(self, rect):
        b = self.bounds()
        inset = 4
        circle = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(inset, inset, b.size.width - 2 * inset, b.size.height - 2 * inset))
        accent = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.85, 0.45, 0.95)
        if self.filled:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.55, 1.0, 0.9).setFill()
            circle.fill()
        circle.setLineWidth_(3)
        accent.setStroke()
        circle.stroke()


class PreviewView(NSView):
    def drawRect_(self, rect):
        b = self.bounds()
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(2, 2, b.size.width - 4, b.size.height - 4), 10, 10)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.65, 1.0, 0.18).setFill()
        path.fill()
        path.setLineWidth_(3)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.65, 1.0, 0.8).setStroke()
        path.stroke()


class FrameView(NSView):
    def drawRect_(self, rect):
        b = self.bounds()
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(5, 5, b.size.width - 10, b.size.height - 10), 14, 14)
        path.setLineWidth_(10)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.25, 0.3, 0.85).setStroke()
        path.stroke()


class CursorOverlay:
    """Shown only for pinch activity: a dot while dragging, and a brief ring where a pinch
    landed on nothing. With `while_armed` it also tracks the hand as a ring while armed."""

    def __init__(self, bus: Bus, while_armed: bool = False, miss_flash_ms: int = 600) -> None:
        self.while_armed = while_armed
        self.miss_flash_ns = miss_flash_ms * 1_000_000
        self._flash_until_ns = 0
        self.screen_h = NSScreen.mainScreen().frame().size.height
        self.cursor = _panel(0, 0, CURSOR_SIZE, CURSOR_SIZE, level_offset=2)
        self.cursor_view = CursorView.alloc().initWithFrame_(NSMakeRect(0, 0, CURSOR_SIZE, CURSOR_SIZE))
        self.cursor.setContentView_(self.cursor_view)
        screen = NSScreen.mainScreen().frame()
        self.frame = _panel(screen.origin.x, screen.origin.y, screen.size.width, screen.size.height, level_offset=3)
        self.frame.setContentView_(FrameView.alloc().initWithFrame_(NSMakeRect(0, 0, screen.size.width, screen.size.height)))
        self.preview = _panel(0, 0, 10, 10, level_offset=1)
        self.preview.setContentView_(PreviewView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)))
        self.anchor = _panel(0, 0, CURSOR_SIZE, CURSOR_SIZE, level_offset=2)  # the sticky scroll anchor
        self.anchor_view = CursorView.alloc().initWithFrame_(NSMakeRect(0, 0, CURSOR_SIZE, CURSOR_SIZE))
        self.anchor.setContentView_(self.anchor_view)
        self.mode = "idle"
        bus.subscribe(ScrollEvent, self._on_scroll)
        bus.subscribe(PointerMoved, self._on_pointer)
        bus.subscribe(ModeChanged, self._on_mode)
        bus.subscribe(SnapPreview, self._on_snap)
        bus.subscribe(DragEvent, self._on_drag)
        bus.subscribe(Tick, self._on_tick)

    def close(self) -> None:
        """Hide every panel; the app drops the bus this overlay listens on when it rebuilds from a new config."""
        for panel in (self.cursor, self.frame, self.preview, self.anchor):
            if panel.isVisible():
                panel.orderOut_(None)

    def _on_mode(self, ev: ModeChanged) -> None:
        self.mode = ev.new
        if ev.new != SCROLL and self.anchor.isVisible():
            self.anchor.orderOut_(None)
        if ev.new not in (ARMED, DRAGGING, SCROLL):
            if self.cursor.isVisible():
                self.cursor.orderOut_(None)
            if self.frame.isVisible():
                self.frame.orderOut_(None)
            if self.preview.isVisible():
                self.preview.orderOut_(None)

    def _on_drag(self, ev: DragEvent) -> None:
        if ev.phase is DragPhase.PAUSE:
            self.frame.orderFrontRegardless()
        elif ev.phase in (DragPhase.RESUME, DragPhase.END) and self.frame.isVisible():
            self.frame.orderOut_(None)
        if ev.phase is DragPhase.MISS:
            self._flash_until_ns = ev.t_ns + self.miss_flash_ns
            self._show(ev.x, ev.y, filled=False)

    def _on_tick(self, ev: Tick) -> None:
        if self._flash_until_ns and ev.t_ns >= self._flash_until_ns and self.mode != DRAGGING:
            self._flash_until_ns = 0
            if self.cursor.isVisible():
                self.cursor.orderOut_(None)

    def _show(self, x: float, y: float, filled: bool) -> None:
        self.cursor_view.filled = filled
        self.cursor.setFrameOrigin_((x - CURSOR_SIZE / 2, self.screen_h - y - CURSOR_SIZE / 2))
        if not self.cursor.isVisible():
            self.cursor.orderFrontRegardless()
        self.cursor_view.setNeedsDisplay_(True)

    def _on_scroll(self, ev: ScrollEvent) -> None:
        if ev.phase is ScrollPhase.END:
            for panel in (self.anchor, self.cursor):
                if panel.isVisible():
                    panel.orderOut_(None)
            return
        if ev.phase is ScrollPhase.SETTLE:
            self._show(ev.x, ev.y, filled=False)  # hollow: hold still here to set the anchor
            return
        if ev.phase is ScrollPhase.START:
            self.anchor_view.filled = False
            self.anchor.setFrameOrigin_((ev.ax - CURSOR_SIZE / 2, self.screen_h - ev.ay - CURSOR_SIZE / 2))
            if not self.anchor.isVisible():
                self.anchor.orderFrontRegardless()
            self.anchor_view.setNeedsDisplay_(True)
        self._show(ev.x, ev.y, filled=True)

    def _on_pointer(self, ev: PointerMoved) -> None:
        if self.mode == DRAGGING:
            self._show(ev.x, ev.y, filled=True)
        elif self.mode == ARMED and self.while_armed:
            self._show(ev.x, ev.y, filled=False)

    def _on_snap(self, ev: SnapPreview) -> None:
        if ev.zone is None:
            if self.preview.isVisible():
                self.preview.orderOut_(None)
            return
        self.preview.setFrame_display_(NSMakeRect(ev.x, self.screen_h - ev.y - ev.h, ev.w, ev.h), True)
        self.preview.contentView().setFrame_(NSMakeRect(0, 0, ev.w, ev.h))
        if not self.preview.isVisible():
            self.preview.orderFrontRegardless()
        self.preview.contentView().setNeedsDisplay_(True)
