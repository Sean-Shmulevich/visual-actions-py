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

from ..core.events import Bus, ModeChanged, PointerMoved, SnapPreview
from ..core.modes import ARMED, DRAGGING

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


class CursorOverlay:
    def __init__(self, bus: Bus) -> None:
        self.screen_h = NSScreen.mainScreen().frame().size.height
        self.cursor = _panel(0, 0, CURSOR_SIZE, CURSOR_SIZE, level_offset=2)
        self.cursor_view = CursorView.alloc().initWithFrame_(NSMakeRect(0, 0, CURSOR_SIZE, CURSOR_SIZE))
        self.cursor.setContentView_(self.cursor_view)
        self.preview = _panel(0, 0, 10, 10, level_offset=1)
        self.preview.setContentView_(PreviewView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)))
        self.mode = "idle"
        bus.subscribe(PointerMoved, self._on_pointer)
        bus.subscribe(ModeChanged, self._on_mode)
        bus.subscribe(SnapPreview, self._on_snap)

    def _on_mode(self, ev: ModeChanged) -> None:
        self.mode = ev.new
        if ev.new not in (ARMED, DRAGGING):
            if self.cursor.isVisible():
                self.cursor.orderOut_(None)
            if self.preview.isVisible():
                self.preview.orderOut_(None)

    def _on_pointer(self, ev: PointerMoved) -> None:
        if self.mode not in (ARMED, DRAGGING):
            return
        self.cursor_view.filled = ev.dragging
        # AppKit origin is bottom-left
        self.cursor.setFrameOrigin_((ev.x - CURSOR_SIZE / 2, self.screen_h - ev.y - CURSOR_SIZE / 2))
        if not self.cursor.isVisible():
            self.cursor.orderFrontRegardless()
        self.cursor_view.setNeedsDisplay_(True)

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
