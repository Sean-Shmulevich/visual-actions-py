from visual_actions.core.automation import Rect, WindowInfo
from visual_actions.platform.macos.windows import match_ax_window, pick_window_at
from visual_actions.platform.mock.automation import MockAutomation


def test_window_at_picks_front_most_in_overlap():
    m = MockAutomation()
    assert m.window_at(500, 300).title == "Front"  # inside both, front wins
    assert m.window_at(150, 150).title == "Back"  # only the back window
    assert m.window_at(5, 5) is None


def test_move_mutates_frame_and_keeps_size():
    m = MockAutomation()
    win = m.window_at(500, 300)
    assert m.move_window(win, 10, 20)
    moved = next(w for w in m.list_windows() if w.id == win.id)
    assert moved.frame == Rect(10, 20, 600, 400)
    assert ("move_window", win.id, 10, 20) in m.calls


def test_unknown_or_unmovable_window_returns_false():
    m = MockAutomation()
    ghost = WindowInfo(id=99, pid=1, app="x", title="x", frame=Rect(0, 0, 10, 10))
    assert m.move_window(ghost, 0, 0) is False
    frozen = MockAutomation(movable=False)
    assert frozen.move_window(frozen.windows[0], 0, 0) is False


def test_pick_window_at_skips_own_process_and_menu_extras():
    wins = [
        WindowInfo(id=1, pid=42, app="me", title="overlay", frame=Rect(0, 0, 2000, 2000)),  # our own
        WindowInfo(id=2, pid=7, app="Extra", title="", frame=Rect(100, 100, 20, 20)),  # tiny, untitled
        WindowInfo(id=3, pid=8, app="Editor", title="main.py", frame=Rect(50, 50, 500, 500)),
    ]
    assert pick_window_at(wins, 110, 110, own_pid=42).id == 3


def test_match_ax_window_by_frame_then_title():
    target = WindowInfo(id=1, pid=1, app="A", title="Doc", frame=Rect(100, 100, 400, 300))
    cands = [(0, "Other", Rect(0, 0, 400, 300)), (1, "Doc", Rect(101, 99, 401, 300))]
    assert match_ax_window(target, cands) == 1  # frame within 2 pt
    cands = [(0, "Doc", Rect(0, 0, 400, 300)), (1, "Other", Rect(900, 900, 10, 10))]
    assert match_ax_window(target, cands) == 0  # unique title
    cands = [(0, "Doc", Rect(0, 0, 1, 1)), (1, "Doc", Rect(5, 5, 1, 1))]
    assert match_ax_window(target, cands) is None  # ambiguous title, no frame match
