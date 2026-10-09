"""The Learning submenu's view-model: status line, titles, schedule status, no rumps."""

from datetime import date
from pathlib import Path

from visual_actions.core.config import default_config
from visual_actions.ui.learning_menu import (
    format_status,
    latest_report,
    learn_settings,
    load_last,
    marks_today,
    misfire_summary,
    nightly_title,
    run_command,
    schedule_installed,
)


def test_status_line_with_a_promoted_run_and_marks():
    last = {"date": "2026-10-09", "decision": "promoted", "model_version": 3, "numbers": {"misfires": {"before": 12, "after": 8}}}
    assert format_status(last, marked_today=2) == "Learning: 2026-10-09 promoted v3, misfires 12 -> 8, 2 marked today"
    assert format_status(last) == "Learning: 2026-10-09 promoted v3, misfires 12 -> 8"


def test_status_line_discarded_with_reason_and_float_rates():
    last = {"date": "2026-10-08T02:00:11", "decision": "discarded", "reason": "accuracy fell 0.03", "misfire_rate": [0.5, 0.75]}
    assert format_status(last) == "Learning: 2026-10-08 discarded, misfires 0.50 -> 0.75, accuracy fell 0.03"


def test_status_line_running_no_run_and_unknown_shapes():
    assert format_status(None) == "Learning: no run yet"
    assert format_status(None, marked_today=1) == "Learning: no run yet, 1 marked today"
    assert format_status({"status": "skipped", "when": "2026-10-07"}, running=True) == "Learning: running…"
    assert format_status({"status": "skipped", "when": "2026-10-07", "version": "user-7"}) == "Learning: 2026-10-07 skipped user-7"
    assert misfire_summary({"numbers": {"accuracy": 0.9}}) is None
    assert misfire_summary({"misfires": 4}) == "misfires 4"
    assert misfire_summary({"metrics": {"misfires": {"champion": 3, "candidate": 1}}}) == "misfires 3 -> 1"


def test_titles_settings_and_command():
    assert nightly_title(2) == "Improve nightly at 02:00"
    assert nightly_title(23) == "Improve nightly at 23:00"
    cfg = default_config()
    enabled, hour = learn_settings(cfg)
    assert isinstance(enabled, bool) and 0 <= hour < 24
    assert learn_settings(object()) == (True, 2)  # no [learn] section yet: defaults
    cmd = run_command()
    assert cmd[1:] == ["-m", "visual_actions.learn", "run"] and cmd[0].endswith("python") or "python" in cmd[0]


def test_schedule_status_shapes():
    assert schedule_installed(True) and not schedule_installed(False)
    assert schedule_installed({"installed": True, "hour": 2}) and not schedule_installed({"installed": False})
    assert schedule_installed({"loaded": True}) and not schedule_installed({})
    assert schedule_installed("installed, next run 02:00") and not schedule_installed("not installed")

    class S:
        installed = True

    assert schedule_installed(S()) and not schedule_installed(None)


def test_last_json_and_latest_report(tmp_path: Path):
    assert load_last(tmp_path / "missing.json") is None
    (tmp_path / "last.json").write_text("{bad json")
    assert load_last(tmp_path / "last.json") is None
    (tmp_path / "last.json").write_text('{"decision": "promoted"}')
    assert load_last(tmp_path / "last.json") == {"decision": "promoted"}
    assert latest_report(tmp_path / "reports") is None
    (tmp_path / "reports").mkdir()
    for name in ("2026-10-07.md", "2026-10-09.md", "2026-10-08.md", "notes.txt"):
        (tmp_path / "reports" / name).write_text("# report")
    assert latest_report(tmp_path / "reports") == tmp_path / "reports" / "2026-10-09.md"


def test_marks_counter_rolls_over_at_midnight():
    assert marks_today(2, date(2026, 10, 9), date(2026, 10, 9)) == (2, date(2026, 10, 9))
    assert marks_today(2, date(2026, 10, 9), date(2026, 10, 10)) == (0, date(2026, 10, 10))
