"""Spike: list windows, find the one at screen centre, nudge it 20 pt and restore it.

    uv run python spikes/spike_windows.py
"""

import time

from visual_actions.platform.macos.automation import MacAutomation


def main() -> int:
    a = MacAutomation()
    w, h = a.screen_size()
    print(f"screen {w}x{h}")
    t0 = time.perf_counter()
    wins = a.list_windows()
    t_list = (time.perf_counter() - t0) * 1000
    print(f"{len(wins)} windows in {t_list:.1f} ms; front 5:")
    for win in wins[:5]:
        print(f"  {win.app!r:28s} {win.title[:30]!r:34s} {win.frame}")

    t0 = time.perf_counter()
    target = a.window_at(w / 2, h / 2)
    t_at = (time.perf_counter() - t0) * 1000
    if target is None:
        print("no window at screen centre")
        return 1
    print(f"window_at centre: {target.app} {target.title!r} {target.frame}  ({t_at:.1f} ms)")

    x0, y0 = target.frame.x, target.frame.y
    times = []
    ok = True
    t0 = time.perf_counter()
    ok = a.move_window(target, x0 + 1, y0 + 1)
    print(f"first move (AX lookup + cache fill): {(time.perf_counter() - t0) * 1000:.1f} ms")
    for i in range(2, 21):  # 20 small moves, like a drag at 30 Hz
        t0 = time.perf_counter()
        ok = a.move_window(target, x0 + i, y0 + i) and ok
        times.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.015)
    time.sleep(0.3)
    t0 = time.perf_counter()
    restored = a.move_window(target, x0, y0)
    t_restore = (time.perf_counter() - t0) * 1000
    time.sleep(0.3)  # the window server reports the new bounds slightly after AX returns
    after = next((x for x in a.list_windows() if x.id == target.id), None)
    times.sort()
    print(f"move_window: first {times and (times[0]):.2f} ms, p50 {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms, restore {t_restore:.2f} ms")
    print(f"moves ok={ok} restored={restored} frame now={after.frame if after else None} expected=({x0}, {y0})")
    return 0 if ok and restored and after and (after.frame.x, after.frame.y) == (x0, y0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
