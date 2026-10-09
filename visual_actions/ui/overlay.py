"""The on-screen HUD: a native vibrancy card at the top centre of the screen.

Main thread only. `OverlayModel` (ui/overlay_model.py) turns the bus events into a `Card`
per tick; this file only renders it with AppKit built in code (no nibs):

  NSPanel (borderless, non-activating, status level, ignores the mouse, all Spaces)
    NSVisualEffectView  HUD material, rounded 14 pt, hairline border, follows dark / light
      NSStackView (horizontal)
        glyph   SF Symbol tinted with the card's accent, ringed by a CAShapeLayer arc
                (the leader hold evidence, or the volume level while adjusting)
        title / subtitle (vertical NSStackView, system fonts)
        trailing label (percent, seconds left, repeat count, snap zone)
      countdown bar  a CALayer along the bottom that drains to the deadline
                     (green while armed, orange in the repeat window, red for the hand-lost grace)

The panel fades in and out with NSAnimationContext; one-shot cards (fired, timed out,
cancelled, dropped, vetoed) hide after popup_ms, the rest stay while their state lasts.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import AppKit
from AppKit import (
    NSAnimationContext,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSImage,
    NSImageView,
    NSInsetRect,
    NSLayoutConstraint,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStackView,
    NSStatusWindowLevel,
    NSTextField,
    NSView,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Quartz import (
    CALayer,
    CAShapeLayer,
    CATransaction,
    CGPathAddArc,
    CGPathCreateMutable,
    kCALineCapRound,
)

from ..core.drag import DragEvent
from ..core.scroll import ScrollEvent
from ..core.events import (
    ActionFired,
    Bus,
    HandLost,
    HandSeen,
    HoldProgress,
    ModeChanged,
    PalmVetoed,
    SnapPreview,
    Tick,
    TokenEmitted,
)
from ..core.modes import ADJUST, IDLE, Timing
from .overlay_model import BLUE, GRAY, GREEN, ORANGE, PURPLE, RED, Card, OverlayModel

W, H = 320, 66
CORNER = 14.0
GLYPH = 40.0  # the glyph box; the ring is drawn just inside it
RING_WIDTH = 3.0
BAR_INSET, BAR_Y, BAR_H = 14.0, 6.0, 3.0
FADE_IN_S, FADE_OUT_S = 0.15, 0.25

# NSVisualEffectView constants, by value so the module imports on older PyObjC builds too.
MATERIAL_HUD = getattr(AppKit, "NSVisualEffectMaterialHUDWindow", 13)
BLEND_BEHIND_WINDOW = getattr(AppKit, "NSVisualEffectBlendingModeBehindWindow", 0)
STATE_ACTIVE = getattr(AppKit, "NSVisualEffectStateActive", 1)
ORIENTATION_HORIZONTAL = getattr(AppKit, "NSUserInterfaceLayoutOrientationHorizontal", 0)
ORIENTATION_VERTICAL = getattr(AppKit, "NSUserInterfaceLayoutOrientationVertical", 1)
DISTRIBUTION_FILL = getattr(AppKit, "NSStackViewDistributionFill", 0)
ALIGN_CENTER_Y = getattr(AppKit, "NSLayoutAttributeCenterY", 10)
ALIGN_LEADING = getattr(AppKit, "NSLayoutAttributeLeading", 5)
WEIGHT_REGULAR = getattr(AppKit, "NSFontWeightRegular", 0.0)
WEIGHT_MEDIUM = getattr(AppKit, "NSFontWeightMedium", 0.23)
WEIGHT_SEMIBOLD = getattr(AppKit, "NSFontWeightSemibold", 0.3)


def accent_color(name: str) -> NSColor:
    return {
        BLUE: NSColor.systemBlueColor(),
        GREEN: NSColor.systemGreenColor(),
        PURPLE: NSColor.systemPurpleColor(),
        RED: NSColor.systemRedColor(),
        ORANGE: NSColor.systemOrangeColor(),
        GRAY: NSColor.systemGrayColor(),
    }.get(name, NSColor.labelColor())


def _label(size: float, weight: float, color: NSColor, mono_digits: bool = False) -> NSTextField:
    field = NSTextField.labelWithString_("")
    font = NSFont.monospacedDigitSystemFontOfSize_weight_(size, weight) if mono_digits else NSFont.systemFontOfSize_weight_(size, weight)
    field.setFont_(font)
    field.setTextColor_(color)
    field.setLineBreakMode_(NSLineBreakByTruncatingTail)
    field.setMaximumNumberOfLines_(1)
    field.setTranslatesAutoresizingMaskIntoConstraints_(False)
    return field


def _ring_path(size: float, inset: float):
    """A full circle starting at 12 o'clock and running clockwise, for strokeEnd = fraction."""
    path = CGPathCreateMutable()
    c, r = size / 2, size / 2 - inset
    CGPathAddArc(path, None, c, c, r, math.pi / 2, math.pi / 2 - 2 * math.pi, True)
    return path


class BorderView(NSView):
    """A hairline rounded border on top of the vibrancy card; `separatorColor` follows the appearance."""

    def drawRect_(self, rect):
        rect = NSInsetRect(self.bounds(), 0.5, 0.5)
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, CORNER - 0.5, CORNER - 0.5)
        path.setLineWidth_(1.0)
        NSColor.separatorColor().setStroke()
        path.stroke()

    def hitTest_(self, point):
        return None


class Overlay:
    def __init__(
        self,
        bus: Bus,
        hold_ns: int,
        timeout_ns: int,
        popup_ms: int,
        grace_ns: int = 2_500_000_000,
        volume: Callable[[], tuple[float, bool] | None] | None = None,
        timing: Timing | None = None,
    ) -> None:
        self.volume = volume  # reads the real system volume; polled only while adjusting
        self._volume_cache: tuple[float, bool] | None = None
        self._volume_read_ns = 0
        self.hold_ns, self.timeout_ns, self.popup_ns = hold_ns, timeout_ns, popup_ms * 1_000_000
        self.grace_ns = grace_ns
        repeat_window_ns = (timing or Timing()).repeat_window_ns
        self.model = OverlayModel(timeout_ns, self.popup_ns, grace_ns, repeat_window_ns)
        self.card: Card | None = None  # what the panel shows right now (None = hidden)
        self._shown = False
        self._fade_gen = 0
        self._symbol_cache: dict[str, NSImage | None] = {}

        self._build_panel()

        bus.subscribe(ModeChanged, self._on_mode)
        bus.subscribe(HoldProgress, self._on_hold)
        bus.subscribe(TokenEmitted, self._on_token)
        bus.subscribe(DragEvent, self._on_drag)
        bus.subscribe(ScrollEvent, self._on_scroll)
        bus.subscribe(ActionFired, self._on_action)
        bus.subscribe(SnapPreview, self._on_snap)
        bus.subscribe(HandLost, self._on_hand_lost)
        bus.subscribe(HandSeen, self._on_hand_seen)
        bus.subscribe(PalmVetoed, self._on_veto)
        bus.subscribe(Tick, self._on_tick)

    # -- construction -------------------------------------------------------

    def _build_panel(self) -> None:
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self.panel.setLevel_(NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHasShadow_(True)
        self.panel.setAlphaValue_(0.0)
        self.panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)
        self._place()

        self.effect = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        self.effect.setMaterial_(MATERIAL_HUD)
        self.effect.setBlendingMode_(BLEND_BEHIND_WINDOW)
        self.effect.setState_(STATE_ACTIVE)
        self.effect.setWantsLayer_(True)
        self.effect.layer().setCornerRadius_(CORNER)
        self.effect.layer().setMasksToBounds_(True)
        self.panel.setContentView_(self.effect)

        # glyph: symbol (or text fallback) inside a ring
        self.glyph_box = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, GLYPH, GLYPH))
        self.glyph_box.setWantsLayer_(True)
        self.glyph_box.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.glyph_box.widthAnchor().constraintEqualToConstant_(GLYPH).setActive_(True)
        self.glyph_box.heightAnchor().constraintEqualToConstant_(GLYPH).setActive_(True)
        self.symbol_view = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, GLYPH, GLYPH))
        self.symbol_view.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.symbol_view.setImageScaling_(getattr(AppKit, "NSImageScaleProportionallyDown", 3))
        self.glyph_text = _label(18, WEIGHT_MEDIUM, NSColor.labelColor())
        self.glyph_text.setAlignment_(getattr(AppKit, "NSTextAlignmentCenter", 1))
        self.glyph_text.setHidden_(True)
        for sub in (self.symbol_view, self.glyph_text):
            self.glyph_box.addSubview_(sub)
            sub.centerXAnchor().constraintEqualToAnchor_(self.glyph_box.centerXAnchor()).setActive_(True)
            sub.centerYAnchor().constraintEqualToAnchor_(self.glyph_box.centerYAnchor()).setActive_(True)
        self.ring_track = CAShapeLayer.layer()
        self.ring = CAShapeLayer.layer()
        for layer in (self.ring_track, self.ring):
            layer.setFrame_(NSMakeRect(0, 0, GLYPH, GLYPH))
            layer.setPath_(_ring_path(GLYPH, RING_WIDTH / 2 + 0.5))
            layer.setFillColor_(None)
            layer.setLineWidth_(RING_WIDTH)
            layer.setLineCap_(kCALineCapRound)
            self.glyph_box.layer().addSublayer_(layer)
        self.ring.setStrokeEnd_(0.0)

        # text column
        self.title = _label(15, WEIGHT_SEMIBOLD, NSColor.labelColor())
        self.subtitle = _label(12, WEIGHT_REGULAR, NSColor.secondaryLabelColor())
        text = NSStackView.stackViewWithViews_([self.title, self.subtitle])
        text.setOrientation_(ORIENTATION_VERTICAL)
        text.setAlignment_(ALIGN_LEADING)
        text.setSpacing_(1)
        text.setTranslatesAutoresizingMaskIntoConstraints_(False)
        text.setContentHuggingPriority_forOrientation_(1, ORIENTATION_HORIZONTAL)
        for field in (self.title, self.subtitle):
            field.setContentCompressionResistancePriority_forOrientation_(250, ORIENTATION_HORIZONTAL)
            field.setContentHuggingPriority_forOrientation_(250, ORIENTATION_HORIZONTAL)

        # trailing figure
        self.trailing = _label(12, WEIGHT_MEDIUM, NSColor.tertiaryLabelColor(), mono_digits=True)
        self.trailing.setAlignment_(getattr(AppKit, "NSTextAlignmentRight", 2))
        self.trailing.setContentCompressionResistancePriority_forOrientation_(750, ORIENTATION_HORIZONTAL)
        self.trailing.setContentHuggingPriority_forOrientation_(750, ORIENTATION_HORIZONTAL)

        row = NSStackView.stackViewWithViews_([self.glyph_box, text, self.trailing])
        row.setOrientation_(ORIENTATION_HORIZONTAL)
        row.setDistribution_(DISTRIBUTION_FILL)  # the text column takes the slack, the figure hugs the trailing edge
        row.setAlignment_(ALIGN_CENTER_Y)
        row.setSpacing_(12)
        row.setEdgeInsets_((10, 14, 14, 14))
        row.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.effect.addSubview_(row)
        NSLayoutConstraint.activateConstraints_(
            [
                row.leadingAnchor().constraintEqualToAnchor_(self.effect.leadingAnchor()),
                row.trailingAnchor().constraintEqualToAnchor_(self.effect.trailingAnchor()),
                row.topAnchor().constraintEqualToAnchor_(self.effect.topAnchor()),
                row.bottomAnchor().constraintEqualToAnchor_(self.effect.bottomAnchor()),
            ]
        )

        # countdown bar along the bottom
        self.bar_track = CALayer.layer()
        self.bar = CALayer.layer()
        for layer in (self.bar_track, self.bar):
            layer.setFrame_(NSMakeRect(BAR_INSET, BAR_Y, W - 2 * BAR_INSET, BAR_H))
            layer.setCornerRadius_(BAR_H / 2)
            layer.setAnchorPoint_((0.0, 0.5))
            layer.setPosition_((BAR_INSET, BAR_Y + BAR_H / 2))
            layer.setHidden_(True)
            self.effect.layer().addSublayer_(layer)

        border = BorderView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        border.setAutoresizingMask_(getattr(AppKit, "NSViewWidthSizable", 2) | getattr(AppKit, "NSViewHeightSizable", 16))
        self.effect.addSubview_(border)

    def _place(self) -> None:
        """Top centre of the main screen, just under the menu bar, like a system HUD."""
        screen = NSScreen.mainScreen()
        if screen is None:
            return
        frame = screen.visibleFrame()
        x = frame.origin.x + (frame.size.width - W) / 2
        y = frame.origin.y + frame.size.height - H - 12
        self.panel.setFrame_display_(NSMakeRect(x, y, W, H), False)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Hide the panel; the app drops the bus this overlay listens on when it rebuilds from a new config."""
        self._shown = False
        self._fade_gen += 1
        if self.panel.isVisible():
            self.panel.orderOut_(None)
        self.panel.setAlphaValue_(0.0)

    # -- events (all on the main thread) ------------------------------------

    def _on_mode(self, ev: ModeChanged) -> None:
        self.model.on_mode(ev)

    def _on_hold(self, ev: HoldProgress) -> None:
        self.model.on_hold(ev)

    def _on_token(self, ev: TokenEmitted) -> None:
        self.model.on_token(ev)

    def _on_drag(self, ev: DragEvent) -> None:
        self.model.on_drag(ev)

    def _on_scroll(self, ev: ScrollEvent) -> None:
        self.model.on_scroll(ev)

    def _on_action(self, ev: ActionFired) -> None:
        self.model.on_action(ev)

    def _on_snap(self, ev: SnapPreview) -> None:
        self.model.on_snap(ev)

    def _on_hand_lost(self, ev: HandLost) -> None:
        self.model.on_hand_lost(ev)

    def _on_hand_seen(self, ev: HandSeen) -> None:
        self.model.on_hand_seen(ev)

    def _on_veto(self, ev: PalmVetoed) -> None:
        self.model.on_veto(ev)

    def _read_volume(self, now: int) -> tuple[float, bool] | None:
        if self.volume is None:
            return None
        if now - self._volume_read_ns >= 50_000_000:  # ~1 ms CoreAudio read, 20 times a second at most
            self._volume_read_ns = now
            try:
                self._volume_cache = self.volume()
            except Exception:  # noqa: BLE001 - a failed read shows "Volume" without a number
                self._volume_cache = None
        return self._volume_cache

    def _on_tick(self, ev: Tick) -> None:
        now = ev.t_ns
        m = self.model
        volume = self._read_volume(now) if m.mode == ADJUST or (m.mode == IDLE and now < m.adjust_until_ns) else None
        self.draw(m.render(now, volume))

    # -- rendering ----------------------------------------------------------

    def draw(self, card: Card | None) -> None:
        """Put `card` on the panel (fading it in if hidden) or fade the panel out for None."""
        if card is None:
            if self.card is not None:
                self.card = None
                self._hide()
            return
        self._apply(card)
        self.card = card
        self._show()

    def _apply(self, card: Card) -> None:
        accent = accent_color(card.accent)
        image = self._symbol(card.symbol, accent)
        if image is not None:
            self.symbol_view.setImage_(image)
            self.symbol_view.setContentTintColor_(accent)
            self.symbol_view.setHidden_(False)
            self.glyph_text.setHidden_(True)
        else:
            self.glyph_text.setStringValue_(card.glyph)
            self.glyph_text.setTextColor_(accent)
            self.glyph_text.setHidden_(False)
            self.symbol_view.setHidden_(True)
        self.title.setStringValue_(card.title)
        self.subtitle.setStringValue_(card.subtitle)
        self.trailing.setStringValue_(card.trailing)
        self.trailing.setHidden_(not card.trailing)

        CATransaction.begin()
        CATransaction.setDisableActions_(True)  # one exact value per tick: the model already extrapolates
        try:
            track = self._cg(NSColor.labelColor().colorWithAlphaComponent_(0.12))
            show_ring = card.progress is not None
            self.ring_track.setHidden_(not show_ring)
            self.ring.setHidden_(not show_ring)
            if show_ring:
                self.ring_track.setStrokeColor_(track)
                self.ring.setStrokeColor_(self._cg(accent))
                self.ring.setStrokeEnd_(max(0.0, min(1.0, card.progress or 0.0)))
            show_bar = card.bar_fraction is not None
            self.bar_track.setHidden_(not show_bar)
            self.bar.setHidden_(not show_bar)
            if show_bar:
                self.bar_track.setBackgroundColor_(track)
                self.bar.setBackgroundColor_(self._cg(accent))
                width = (W - 2 * BAR_INSET) * max(0.0, min(1.0, card.bar_fraction or 0.0))
                self.bar.setBounds_(NSMakeRect(0, 0, width, BAR_H))
        finally:
            CATransaction.commit()

    def _symbol(self, name: str, accent: NSColor) -> NSImage | None:
        if name not in self._symbol_cache:
            image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
            if image is not None and hasattr(AppKit, "NSImageSymbolConfiguration"):
                cfg = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(20, WEIGHT_MEDIUM)
                configured = image.imageWithSymbolConfiguration_(cfg)
                image = configured or image
            if image is not None:
                image.setTemplate_(True)
            self._symbol_cache[name] = image
        return self._symbol_cache[name]

    def _cg(self, color: NSColor):
        """A CGColor resolved under the panel's appearance, so dynamic colours follow dark / light."""
        appearance = self.effect.effectiveAppearance()
        if appearance is not None and hasattr(appearance, "performAsCurrentDrawingAppearance_"):
            out: list = []
            appearance.performAsCurrentDrawingAppearance_(lambda: out.append(color.CGColor()))
            if out:
                return out[0]
        return color.CGColor()

    def _show(self) -> None:
        if self._shown:
            return
        self._shown = True
        self._fade_gen += 1
        if not self.panel.isVisible():
            self._place()
            self.panel.setAlphaValue_(0.0)
            self.panel.orderFrontRegardless()
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(FADE_IN_S)
        self.panel.animator().setAlphaValue_(1.0)
        NSAnimationContext.endGrouping()

    def _hide(self) -> None:
        if not self._shown:
            return
        self._shown = False
        self._fade_gen += 1
        gen = self._fade_gen

        def fade(ctx):
            ctx.setDuration_(FADE_OUT_S)
            self.panel.animator().setAlphaValue_(0.0)

        def done():
            if gen == self._fade_gen and not self._shown and self.panel.isVisible():
                self.panel.orderOut_(None)

        NSAnimationContext.runAnimationGroup_completionHandler_(fade, done)
