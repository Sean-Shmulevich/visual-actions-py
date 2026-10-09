"""Native settings window: one tab per config section, built in code from the view-model.

Main thread only (rumps owns the NSApplication loop; the menu callback that opens the
window runs there). The window is created once and hidden on close, so it survives
being closed and reopened. No modal dialogs: errors go to the status line.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSButton,
    NSButtonTypeMomentaryPushIn,
    NSButtonTypeSwitch,
    NSColor,
    NSControlStateValueOn,
    NSFont,
    NSLocale,
    NSMakeRect,
    NSNumberFormatter,
    NSNumberFormatterDecimalStyle,
    NSPopUpButton,
    NSScrollView,
    NSSlider,
    NSTabView,
    NSTabViewItem,
    NSTextAlignmentRight,
    NSTextField,
    NSView,
    NSViewHeightSizable,
    NSViewWidthSizable,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject

from ..core.config import Config
from ..paths import config_path
from .settings_model import RESTART_NOTE, SECTION_TITLES, FieldSpec, SettingsModel, format_value

W, H = 620, 640
ROW_H = 30
LABEL_W = 210
CONTROL_X = LABEL_W + 20
PAD = 12


class FlippedView(NSView):
    """Rows are laid out top-down; a flipped view makes y grow downwards like a form."""

    def isFlipped(self):
        return True


class _Target(NSObject):
    """Objective-C side of the controls: every action and delegate call forwards to the
    Python `SettingsWindow` so the logic stays in one place."""

    owner = objc.ivar()

    def fieldChanged_(self, sender) -> None:
        self.owner.on_text(str(sender.identifier()), str(sender.stringValue()))

    def controlTextDidChange_(self, note) -> None:
        field = note.object()
        self.owner.on_text(str(field.identifier()), str(field.stringValue()))

    @objc.typedSelector(b"Z@:@@@")
    def control_didFailToFormatString_errorDescription_(self, control, string, error) -> bool:
        return True  # keep the text; the view-model reports the problem in the status line, no alert

    def checkChanged_(self, sender) -> None:
        self.owner.on_bool(str(sender.identifier()), sender.state() == NSControlStateValueOn)

    def sliderChanged_(self, sender) -> None:
        self.owner.on_slider(str(sender.identifier()), float(sender.doubleValue()))

    def popupChanged_(self, sender) -> None:
        self.owner.on_choice(str(sender.identifier()), str(sender.titleOfSelectedItem()))

    def save_(self, _sender) -> None:
        self.owner.save()

    def revert_(self, _sender) -> None:
        self.owner.revert()

    def reset_(self, _sender) -> None:
        self.owner.reset()


def _label(text: str, x: float, y: float, w: float, h: float = 22, align_right: bool = False, small: bool = False) -> NSTextField:
    lbl = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    lbl.setStringValue_(text)
    lbl.setBezeled_(False)
    lbl.setDrawsBackground_(False)
    lbl.setEditable_(False)
    lbl.setSelectable_(False)
    if align_right:
        lbl.setAlignment_(NSTextAlignmentRight)
    if small:
        lbl.setFont_(NSFont.systemFontOfSize_(11))
        lbl.setTextColor_(NSColor.secondaryLabelColor())
    return lbl


def _number_formatter(spec: FieldSpec) -> NSNumberFormatter:
    f = NSNumberFormatter.alloc().init()
    f.setNumberStyle_(NSNumberFormatterDecimalStyle)
    f.setLocale_(NSLocale.localeWithLocaleIdentifier_("en_US_POSIX"))
    f.setUsesGroupingSeparator_(False)
    f.setMinimumFractionDigits_(0)
    f.setMaximumFractionDigits_(0 if spec.kind == "int" else 4)
    f.setAllowsFloats_(spec.kind != "int")
    return f


class SettingsWindow:
    def __init__(self, model: SettingsModel, on_save: Callable[[Config], None], path: Path | None = None) -> None:
        self.model = model
        self.on_save = on_save
        self.path = path or config_path()
        self.controls: dict[str, Any] = {}  # key -> NSTextField | NSButton | NSPopUpButton
        self.sliders: dict[str, Any] = {}
        self.target = _Target.alloc().init()
        self.target.owner = self

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("Visual Actions Settings")
        self.window.setReleasedWhenClosed_(False)  # hide on close; show() brings it back
        self.window.center()
        content = self.window.contentView()

        self.tabs = NSTabView.alloc().initWithFrame_(NSMakeRect(PAD, 118, W - 2 * PAD, H - 118 - PAD))
        for section in model.sections():
            item = NSTabViewItem.alloc().initWithIdentifier_(section)
            item.setLabel_(SECTION_TITLES.get(section, section.title()))
            item.setView_(self._section_view(section))
            self.tabs.addTabViewItem_(item)
        content.addSubview_(self.tabs)

        self.note = _label(RESTART_NOTE, PAD, 70, W - 2 * PAD, 40, small=True)
        self.note.cell().setWraps_(True)
        content.addSubview_(self.note)
        self.file_label = _label(f"Config file: {self.path}", PAD, 50, W - 2 * PAD, 18, small=True)
        content.addSubview_(self.file_label)
        self.status = _label("", PAD, 28, W - 2 * PAD, 18)
        content.addSubview_(self.status)

        self.save_button = self._button("Save", W - PAD - 90, 90, "save:")
        self.save_button.setKeyEquivalent_("\r")
        self.revert_button = self._button("Revert", W - PAD - 190, 90, "revert:")
        self.reset_button = self._button("Reset to profile defaults", PAD, 200, "reset:")
        for b in (self.save_button, self.revert_button, self.reset_button):
            content.addSubview_(b)
        self.refresh()

    # -- building ----------------------------------------------------------------

    def _button(self, title: str, x: float, w: float, action: str) -> NSButton:
        b = NSButton.alloc().initWithFrame_(NSMakeRect(x, PAD - 4, w, 32))
        b.setButtonType_(NSButtonTypeMomentaryPushIn)
        b.setBezelStyle_(1)  # rounded
        b.setTitle_(title)
        b.setTarget_(self.target)
        b.setAction_(action)
        return b

    def _section_view(self, section: str) -> NSScrollView:
        specs = self.model.fields_in(section)
        width = W - 2 * PAD - 24
        doc = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, width, ROW_H * len(specs) + PAD))
        y = PAD / 2
        for spec in specs:
            lbl = _label(spec.label, 0, y + 3, LABEL_W, align_right=True)
            lbl.setToolTip_(spec.help or spec.key)
            doc.addSubview_(lbl)
            control = self._control(spec, y)
            control.setIdentifier_(spec.key)
            control.setToolTip_(spec.help or spec.key)
            doc.addSubview_(control)
            self.controls[spec.key] = control
            if spec.numeric and spec.range is not None:
                lo, hi = spec.range
                slider = NSSlider.alloc().initWithFrame_(NSMakeRect(CONTROL_X + 100, y + 2, width - CONTROL_X - 110, 22))
                slider.setMinValue_(lo)
                slider.setMaxValue_(hi)
                slider.setContinuous_(True)
                slider.setIdentifier_(spec.key)
                slider.setToolTip_(spec.help or spec.key)
                slider.setTarget_(self.target)
                slider.setAction_("sliderChanged:")
                doc.addSubview_(slider)
                self.sliders[spec.key] = slider
            y += ROW_H
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, width, 100))
        scroll.setHasVerticalScroller_(True)
        scroll.setDrawsBackground_(False)
        scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        scroll.setDocumentView_(doc)
        return scroll

    def _control(self, spec: FieldSpec, y: float) -> Any:
        if spec.kind == "bool":
            b = NSButton.alloc().initWithFrame_(NSMakeRect(CONTROL_X, y + 3, 24, 22))
            b.setButtonType_(NSButtonTypeSwitch)
            b.setTitle_("")
            b.setTarget_(self.target)
            b.setAction_("checkChanged:")
            return b
        if spec.kind == "choice":
            p = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(CONTROL_X, y, 160, 26), False)
            for c in spec.choices:
                p.addItemWithTitle_(c)
            p.setTarget_(self.target)
            p.setAction_("popupChanged:")
            return p
        w = 90 if spec.numeric else 320
        f = NSTextField.alloc().initWithFrame_(NSMakeRect(CONTROL_X, y + 2, w, 22))
        if spec.numeric:
            f.setFormatter_(_number_formatter(spec))
        f.setDelegate_(self.target)
        f.setTarget_(self.target)
        f.setAction_("fieldChanged:")
        return f

    # -- model -> controls ------------------------------------------------------------

    def refresh(self) -> None:
        for spec in self.model.specs:
            self._refresh_one(spec)
        self._update_status()

    def _refresh_one(self, spec: FieldSpec) -> None:
        control = self.controls[spec.key]
        value = self.model.value(spec.key)
        if spec.kind == "bool":
            control.setState_(1 if value else 0)
        elif spec.kind == "choice":
            control.selectItemWithTitle_(str(value))
        else:
            control.setStringValue_(format_value(spec, value))
            control.setTextColor_(NSColor.textColor())
        slider = self.sliders.get(spec.key)
        if slider is not None and isinstance(value, (int, float)):
            slider.setDoubleValue_(float(value))

    def _update_status(self, message: str | None = None, error: bool = False) -> None:
        if message is None:
            problems = self.model.validate()
            if problems:
                message, error = problems[0], True
            else:
                n = len(self.model.diff())
                message = f"{n} key{'s differ' if n != 1 else ' differs'} from the {self.model.profile} profile"
                if self.model.dirty():
                    message += " (unsaved)"
        self.status.setStringValue_(message)
        self.status.setTextColor_(NSColor.systemRedColor() if error else NSColor.labelColor())

    # -- controls -> model (called by _Target on the main thread) --------------------------

    def on_text(self, key: str, text: str) -> None:
        err = self.model.set_text(key, text)
        self.controls[key].setTextColor_(NSColor.systemRedColor() if err else NSColor.textColor())
        slider = self.sliders.get(key)
        if slider is not None and err is None:
            slider.setDoubleValue_(float(self.model.value(key)))
        if err:
            self._update_status(err, error=True)
        else:
            self._update_status()

    def on_bool(self, key: str, value: bool) -> None:
        self.model.set_value(key, value)
        self._update_status()

    def on_slider(self, key: str, value: float) -> None:
        spec = self.model.spec(key)
        v: int | float = round(value) if spec.kind == "int" else round(value, 3)
        self.model.set_value(key, v)
        self.controls[key].setStringValue_(format_value(spec, v))
        self.controls[key].setTextColor_(NSColor.textColor())
        self._update_status()

    def on_choice(self, key: str, value: str) -> None:
        if key == "leader.profile":
            changed = [k for k in self.model.set_profile(value) if k != key]
            self.refresh()
            self._update_status(f"{value} profile: refilled {', '.join(k.split('.', 1)[1] for k in changed) or 'nothing'}")
        else:
            self.model.set_value(key, value)
            self._update_status()

    def save(self) -> None:
        self.window.makeFirstResponder_(None)  # commit the field being edited
        problems = self.model.validate()
        if problems:
            self._update_status("Not saved: " + problems[0], error=True)
            return
        try:
            cfg = self.model.save(self.path)
        except (OSError, ValueError) as exc:
            self._update_status(f"Not saved: {exc}", error=True)
            return
        try:
            self.on_save(cfg)
        except Exception as exc:  # noqa: BLE001 - the file is written; say what the restart hit
            self.refresh()
            self._update_status(f"Saved; restart failed: {exc}", error=True)
            return
        self.refresh()
        self._update_status(f"Saved to {self.path.name}; pipeline restarted with the new config")

    def revert(self) -> None:
        self.model.revert()
        self.refresh()
        self._update_status("Reverted to the saved config")

    def reset(self) -> None:
        self.model.reset_to_profile_defaults()
        self.refresh()
        self._update_status(f"Reset to the {self.model.profile} profile defaults (unsaved)")

    # -- lifecycle ---------------------------------------------------------------------

    def show(self) -> None:
        NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def close(self) -> None:
        self.window.orderOut_(None)
