"""The HUD view-model: events in, one Card per tick out. Pure Python, no AppKit."""

from visual_actions.core.drag import DragEvent, DragPhase
from visual_actions.core.events import (
    ActionFired,
    HandLost,
    HoldProgress,
    ModeChanged,
    PalmVetoed,
    SnapPreview,
    TokenEmitted,
)
from visual_actions.core.modes import ADJUST, ARMED, DRAGGING, HOLDING, IDLE, REPEAT
from visual_actions.core.types import Action, ActionKind, Hand, Token
from visual_actions.ui.overlay_model import (
    BLUE,
    FALLBACK_GLYPHS,
    GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    OverlayModel,
)

S = 1_000_000_000
TIMEOUT, POPUP, GRACE, REPEAT_WINDOW = 5 * S, 900_000_000, 2_500_000_000, 1_500_000_000


def model() -> OverlayModel:
    return OverlayModel(TIMEOUT, POPUP, GRACE, REPEAT_WINDOW)


def action(name: str = "Cmd+Tab") -> Action:
    return Action(ActionKind.KEY, name)


def test_idle_shows_nothing():
    assert model().render(0) is None


def test_hold_progress_extrapolates_at_the_token_rate():
    m = model()
    m.on_mode(ModeChanged(0, IDLE, HOLDING, namespace="window"))
    m.on_hold(HoldProgress(1 * S, 0.4, rate=1.0))
    card = m.render(1 * S)
    assert card is not None and card.state == "hold" and card.persistent
    assert card.symbol == "hand.raised" and card.accent == BLUE
    assert card.progress == 0.4 and card.trailing == "40%" and card.bar_fraction is None
    assert "palm" in card.title and card.subtitle == "Window menu"
    # between tokens the ring keeps moving at `rate`, capped at 1
    later = m.render(1 * S + 300_000_000)
    assert later is not None and abs(later.progress - 0.7) < 1e-9
    assert m.render(3 * S).progress == 1.0
    # a negative rate (draining) never extrapolates below the reported fraction
    m.on_hold(HoldProgress(2 * S, 0.5, rate=-1.0))
    assert m.render(2 * S + 500_000_000).progress == 0.5


def test_hold_ring_restarts_on_a_namespace_switch_and_media_names_the_peace_sign():
    m = model()
    m.on_mode(ModeChanged(0, IDLE, HOLDING, namespace="window"))
    m.on_hold(HoldProgress(S, 0.6))
    m.on_mode(ModeChanged(S, HOLDING, HOLDING, namespace="media"))
    card = m.render(S)
    assert card.progress == 0.0 and "peace" in card.title and card.subtitle == "Media menu"


def test_armed_counts_down_to_the_deadline_with_the_namespace_symbol():
    m = model()
    m.on_mode(ModeChanged(0, IDLE, HOLDING, namespace="window"))
    m.on_token(TokenEmitted(Token(S, "open_palm", 1.0, Hand.RIGHT, True)))
    m.on_mode(ModeChanged(2 * S, HOLDING, ARMED, namespace="window", deadline_ns=2 * S + TIMEOUT))
    card = m.render(2 * S)
    assert card.state == "armed" and card.persistent and card.symbol == "macwindow" and card.accent == GREEN
    assert card.title == "Window menu" and card.subtitle == "show a gesture"  # the arming palm is not a command
    assert card.bar_fraction == 1.0 and card.trailing == "5.0 s" and card.progress is None
    m.on_token(TokenEmitted(Token(3 * S, "h_left", 0.9, Hand.RIGHT, False)))
    card = m.render(2 * S + 4 * S)
    assert abs(card.bar_fraction - 0.2) < 1e-9 and card.trailing == "1.0 s" and card.subtitle == "last: h_left"
    m.on_mode(ModeChanged(2 * S, HOLDING, ARMED, namespace="media", deadline_ns=2 * S + TIMEOUT))
    assert m.render(2 * S).symbol == "music.note"


def test_armed_survives_a_hand_loss_as_a_red_pause():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="window", deadline_ns=TIMEOUT))
    m.on_hand_lost(HandLost(S))
    card = m.render(S)
    assert card.state == "armed_lost" and card.symbol == "pause.circle" and card.accent == RED
    assert abs(card.bar_fraction - 0.8) < 1e-9
    from visual_actions.core.events import HandSeen

    m.on_hand_seen(HandSeen(hand_frame=None))  # type: ignore[arg-type] - the model only needs the event type
    assert m.render(S).state == "armed"


def test_fired_flashes_over_the_menu_then_auto_hides():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="window", deadline_ns=TIMEOUT))
    m.on_action(ActionFired(S, action("Cmd+Tab"), ok=True))
    card = m.render(S)
    assert card.state == "fired" and not card.persistent
    assert card.symbol == "checkmark.circle" and card.title == "Cmd+Tab" and card.accent == GREEN
    assert m.render(S + POPUP - 1).state == "fired"
    assert m.render(S + POPUP).state == "armed"  # chaining: the menu is still open underneath
    # with chaining off the menu closed at the same instant: that is not a cancel
    m.on_action(ActionFired(2 * S, action("Play"), ok=True))
    m.on_mode(ModeChanged(2 * S, ARMED, IDLE))
    assert m.render(2 * S).state == "fired"
    assert m.render(2 * S + POPUP) is None


def test_failed_action_is_red_with_the_message():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="window", deadline_ns=TIMEOUT))
    m.on_action(ActionFired(S, action("Focus"), ok=False, message="no window"))
    card = m.render(S)
    assert card.state == "failed" and card.accent == RED and card.symbol == "exclamationmark.triangle"
    assert card.title == "Focus" and card.subtitle == "no window"


def test_timeout_and_cancel_are_one_shots():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="window", deadline_ns=TIMEOUT))
    m.on_mode(ModeChanged(TIMEOUT, ARMED, IDLE))
    card = m.render(TIMEOUT)
    assert card.state == "timeout" and card.symbol == "clock" and card.accent == GRAY and not card.persistent
    assert card.subtitle == "Window menu closed"
    assert m.render(TIMEOUT + POPUP) is None

    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="media", deadline_ns=TIMEOUT))
    m.on_mode(ModeChanged(S, ARMED, IDLE))  # fist before the deadline
    card = m.render(S)
    assert card.state == "cancelled" and card.symbol == "xmark.circle" and card.subtitle == "Media menu closed"
    # a broken hold is silent
    m.on_mode(ModeChanged(3 * S, IDLE, HOLDING, namespace="window"))
    m.on_mode(ModeChanged(4 * S, HOLDING, IDLE))
    assert m.render(4 * S) is None


def test_drag_pause_shows_the_grace_draining_in_red_then_resumes():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="window", deadline_ns=TIMEOUT))
    m.on_mode(ModeChanged(S, ARMED, DRAGGING, namespace="window"))
    m.on_drag(DragEvent(S, DragPhase.START, "Safari", 100, 100))
    card = m.render(S)
    assert card.state == "drag" and card.symbol == "hand.point.up" and card.accent == PURPLE
    assert card.subtitle == "Safari" and card.persistent and card.bar_fraction is None
    m.on_snap(SnapPreview(S, "left_half"))
    assert m.render(S).trailing == "left half"
    m.on_drag(DragEvent(2 * S, DragPhase.PAUSE, "Safari", 100, 100))
    card = m.render(2 * S + S)
    assert card.state == "drag_lost" and card.symbol == "pause.circle" and card.accent == RED
    assert abs(card.bar_fraction - 0.6) < 1e-9 and card.trailing == "1.5 s" and "Safari" in card.subtitle
    m.on_drag(DragEvent(3 * S, DragPhase.RESUME, "Safari", 120, 100))
    assert m.render(3 * S).state == "drag"
    m.on_drag(DragEvent(4 * S, DragPhase.END, "Safari", 120, 100, snapped="right_half"))
    m.on_mode(ModeChanged(4 * S, DRAGGING, ARMED, namespace="window", deadline_ns=4 * S + TIMEOUT))
    card = m.render(4 * S)
    assert card.state == "dropped" and card.title == "Snapped right half" and card.subtitle == "Safari"
    assert m.render(4 * S + POPUP).state == "armed"


def test_a_missed_pinch_is_a_gray_one_shot():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="window", deadline_ns=TIMEOUT))
    m.on_drag(DragEvent(S, DragPhase.MISS, "", 10, 10))
    assert m.render(S).state == "missed"
    assert m.render(S + POPUP).state == "armed"


def test_adjust_prompts_until_settled_then_shows_the_live_volume():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="media", deadline_ns=TIMEOUT))
    m.on_mode(ModeChanged(S, ARMED, ADJUST, namespace="media"))
    card = m.render(S, volume=(0.62, False))
    assert card.state == "adjust" and card.symbol == "speaker.wave.2" and card.persistent
    assert card.title == "Volume 62%" and card.subtitle == "hold the pinch still…" and card.progress == 0.62
    assert card.trailing == ""  # the title carries the level
    m.on_mode(ModeChanged(S + 250_000_000, ADJUST, ADJUST, namespace="media"))  # settled
    assert m.render(S + 250_000_000, volume=(0.62, False)).subtitle == "◀ quieter · pinch · louder ▶"
    # volume steps do not flash over the level
    m.on_action(ActionFired(2 * S, action("Volume Up"), ok=True))
    card = m.render(2 * S, volume=(0.68, False))
    assert card.state == "adjust" and card.title == "Volume 68%"
    muted = m.render(2 * S, volume=(0.0, True))
    assert muted.symbol == "speaker.slash" and muted.trailing == "muted" and muted.title == "Volume 0% (muted)"
    assert m.render(2 * S, volume=None).title == "Volume"
    # release: "done" stays for popup_ms, then nothing
    m.on_mode(ModeChanged(3 * S, ADJUST, IDLE))
    card = m.render(3 * S + POPUP - 1, volume=(0.68, False))
    assert card.state == "adjust" and card.subtitle == "done" and not card.persistent
    assert m.render(3 * S + POPUP) is None


def test_repeat_counts_fires_and_drains_the_window():
    m = model()
    m.on_mode(ModeChanged(0, HOLDING, ARMED, namespace="media", deadline_ns=TIMEOUT))
    m.on_action(ActionFired(S, action("Next Track"), ok=True))
    m.on_mode(ModeChanged(S, ARMED, REPEAT, namespace="media", deadline_ns=S + REPEAT_WINDOW))
    card = m.render(S)
    assert card.state == "repeat" and card.symbol == "arrow.left.and.right" and card.accent == ORANGE
    assert card.title == "Next Track" and card.trailing == "×1" and card.bar_fraction == 1.0 and card.persistent
    m.on_action(ActionFired(2 * S, action("Next Track"), ok=True))
    m.on_mode(ModeChanged(2 * S, REPEAT, REPEAT, namespace="media", deadline_ns=2 * S + REPEAT_WINDOW))
    card = m.render(2 * S + 750_000_000)
    assert card.trailing == "×2" and abs(card.bar_fraction - 0.5) < 1e-9
    m.on_mode(ModeChanged(2 * S + REPEAT_WINDOW, REPEAT, ARMED, namespace="media", deadline_ns=10 * S))
    assert m.render(2 * S + REPEAT_WINDOW).state == "armed"


def test_veto_shows_only_while_idle():
    m = model()
    m.on_veto(PalmVetoed(S, overlap=0.5, spread=0.3))
    card = m.render(S)
    assert card.state == "veto" and card.symbol == "hand.raised.slash" and card.accent == ORANGE
    assert m.render(S + POPUP) is None
    m.on_mode(ModeChanged(2 * S, HOLDING, ARMED, namespace="window", deadline_ns=10 * S))
    m.on_veto(PalmVetoed(3 * S, overlap=0.5, spread=0.3))
    assert m.render(3 * S).state == "armed"


def test_every_symbol_has_a_text_fallback():
    m = model()
    m.on_mode(ModeChanged(0, IDLE, HOLDING, namespace="window"))
    assert m.render(0).glyph == FALLBACK_GLYPHS["hand.raised"]
    assert "hand.point.up" in FALLBACK_GLYPHS and "music.note" in FALLBACK_GLYPHS
