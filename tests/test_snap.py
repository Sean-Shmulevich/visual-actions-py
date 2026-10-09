"""Snap zones, dwell, hysteresis; and snapping through the drag controller."""

from visual_actions.core.automation import Rect
from visual_actions.core.drag import DragController, DragEvent, DragPhase, FakeWindows
from visual_actions.core.events import Bus, SnapPreview
from visual_actions.core.pinch import PinchEvent, PinchPhase
from visual_actions.core.pointer import PointerMap, ReachBox, SmoothedPointer
from visual_actions.core.snap import SnapEngine, SnapRules

MS = 1_000_000
VIS = Rect(0, 25, 1440, 875)
RULES = SnapRules(edge_px=28, corner_px=110, dwell_ns=150 * MS, quarters=True, maximize=True)


def engine():
    return SnapEngine(VIS, RULES)


def test_zone_geometry():
    e = engine()
    assert e.zone_at(10, 400).name == "left" and e.zone_at(10, 400).target == Rect(0, 25, 720, 875)
    assert e.zone_at(1435, 400).name == "right" and e.zone_at(1435, 400).target == Rect(720, 25, 720, 875)
    assert e.zone_at(700, 30).name == "top" and e.zone_at(700, 30).target == VIS
    assert e.zone_at(50, 60).name == "top_left" and e.zone_at(50, 60).target == Rect(0, 25, 720, 437)
    assert e.zone_at(1400, 880).name == "bottom_right" and e.zone_at(1400, 880).target == Rect(720, 462, 720, 438)
    assert e.zone_at(700, 450) is None
    assert e.zone_at(700, 895) is None  # bottom edge alone is not a zone


def test_quarters_and_maximize_can_be_disabled():
    e = SnapEngine(VIS, SnapRules(quarters=False, maximize=False))
    assert e.zone_at(10, 60).name == "left"  # corner falls back to the edge zone
    assert e.zone_at(700, 30) is None


def test_dwell_before_preview():
    e = engine()
    assert e.update(10, 400, 0) is None
    assert e.update(10, 400, 100 * MS) is None
    z = e.update(10, 400, 160 * MS)
    assert z is not None and z.name == "left"


def test_sweeping_through_a_zone_never_previews():
    e = engine()
    assert e.update(10, 400, 0) is None
    assert e.update(300, 400, 50 * MS) is None
    assert e.update(10, 400, 100 * MS) is None  # dwell restarted
    assert e.update(10, 400, 200 * MS) is None
    assert e.update(10, 400, 260 * MS) is not None


def test_hysteresis_keeps_zone_until_well_outside():
    e = engine()
    e.update(10, 400, 0)
    assert e.update(10, 400, 200 * MS) is not None
    assert e.update(40, 400, 210 * MS) is not None  # 40 px: outside enter band (28), inside leave band (56)
    assert e.update(80, 400, 220 * MS) is None  # beyond the leave band
    assert e.active is None


def test_switching_zones_requires_new_dwell():
    e = engine()
    e.update(10, 400, 0)
    assert e.update(10, 400, 200 * MS).name == "left"
    assert e.update(50, 60, 210 * MS) is None  # moved to the top-left corner: previous zone dropped
    assert e.update(50, 60, 400 * MS).name == "top_left"


# -- through the drag controller ------------------------------------------------


def pointer():
    return SmoothedPointer(PointerMap(1440, 900, ReachBox(0, 1, 0, 1)), min_cutoff=1e9, beta=0.0)


def pev(phase, t, x, y):
    return PinchEvent(t, phase, x, y, 0.12, 0.2 if phase is not PinchPhase.END else 0.8)


def make():
    bus = Bus()
    wins = FakeWindows([("Win", 400, 300, 500, 300)])
    previews, drags = [], []
    bus.subscribe(SnapPreview, previews.append)
    bus.subscribe(DragEvent, drags.append)
    c = DragController(bus, wins, pointer(), snap=engine())
    return c, wins, previews, drags


def drag_to_right_edge(c, t0=0):
    assert c.on_pinch(pev(PinchPhase.START, t0, 0.5, 0.5))  # (720, 450) inside Win
    for i in range(1, 12):  # sweep right over ~360 ms, ending at x=1435
        x = 0.5 + 0.497 * min(1.0, i / 6)
        c.on_pinch(pev(PinchPhase.MOVE, t0 + i * 33 * MS, x, 0.5))


def test_release_in_zone_snaps_and_previews():
    c, wins, previews, drags = make()
    drag_to_right_edge(c)
    assert previews and previews[-1].zone == "right"
    c.on_pinch(pev(PinchPhase.END, 400 * MS, 0.997, 0.5))
    c.on_tick(900 * MS)
    w = wins.windows[0]
    assert (w["x"], w["y"], w["w"], w["h"]) == (720, 25, 720, 875)
    assert drags[-1].phase is DragPhase.END and drags[-1].snapped == "right"
    assert previews[-1].zone is None  # preview cleared on release


def test_release_outside_zone_just_drops():
    c, wins, previews, drags = make()
    c.on_pinch(pev(PinchPhase.START, 0, 0.5, 0.5))
    c.on_pinch(pev(PinchPhase.MOVE, 33 * MS, 0.55, 0.55))
    c.on_pinch(pev(PinchPhase.END, 66 * MS, 0.55, 0.55))
    c.on_tick(566 * MS)
    w = wins.windows[0]
    assert (w["w"], w["h"]) == (500, 300) and drags[-1].snapped is None
    assert all(p.zone is None for p in previews)


def test_redrag_of_snapped_window_restores_its_size():
    c, wins, previews, drags = make()
    drag_to_right_edge(c)
    c.on_pinch(pev(PinchPhase.END, 400 * MS, 0.997, 0.5))
    c.on_tick(900 * MS)
    assert wins.windows[0]["w"] == 720
    # grab it again in the middle of the snapped half and move a little
    assert c.on_pinch(pev(PinchPhase.START, 1000 * MS, 0.75, 0.5))
    w = wins.windows[0]
    assert (w["w"], w["h"]) == (500, 300)  # original size back
    assert w["x"] <= 1080 <= w["x"] + w["w"] and w["y"] <= 450 <= w["y"] + w["h"]  # pointer still inside
    c.on_pinch(pev(PinchPhase.MOVE, 1033 * MS, 0.70, 0.5))
    c.on_pinch(pev(PinchPhase.END, 1066 * MS, 0.70, 0.5))
    c.on_tick(1566 * MS)
    assert drags[-1].snapped is None and wins.windows[0]["w"] == 500


def test_hand_lost_never_snaps():
    c, wins, previews, drags = make()
    drag_to_right_edge(c)
    assert previews[-1].zone == "right"
    c.cancel(500 * MS)
    assert wins.windows[0]["w"] == 500 and drags[-1].snapped is None and previews[-1].zone is None
