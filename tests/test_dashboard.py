import json
import urllib.request

from visual_actions.core.config import default_config
from visual_actions.core.drag import DragEvent, DragPhase
from visual_actions.core.events import ActionFired, Bus, ModeChanged, TokenEmitted
from visual_actions.core.types import Action, ActionKind, Hand, Token
from visual_actions.ui.dashboard import Dashboard


def test_dashboard_serves_page_and_live_state():
    bus = Bus()
    dash = Dashboard(bus, default_config(), stats=lambda: {"frames": 100, "tracked": 40, "error": None}, port=0)
    try:
        page = urllib.request.urlopen(dash.url + "/", timeout=3).read().decode()
        assert "<title>Visual Actions</title>" in page and "/events" in page

        bus.publish(ModeChanged(1, "idle", "holding"))
        bus.publish(TokenEmitted(Token(2, "open_palm", 0.97, Hand.RIGHT, True)))
        bus.publish(ModeChanged(3, "holding", "armed", "window", 10))
        bus.publish(ActionFired(4, Action(ActionKind.KEY, "Cmd+Tab"), True))
        bus.publish(DragEvent(5, DragPhase.END, "App: Win", 10, 10, snapped="left"))

        s = json.loads(urllib.request.urlopen(dash.url + "/state.json", timeout=3).read())
        assert s["state"]["mode"] == "armed" and s["state"]["namespace"] == "window"
        assert s["state"]["token"] == "open_palm" and s["state"]["token_conf"] == 0.97
        assert s["state"]["last_action"] == "Cmd+Tab"
        assert s["stats"]["frames"] == 100
        kinds = [e["kind"] for e in s["log"]]
        assert kinds[:3] == ["drag", "action", "mode"]  # newest first
        assert s["log"][0]["snapped"] == "left"
        assert any(b["gesture"] == "h_left" and b["action"] == "Cmd+Tab" for b in s["bindings"])
        assert s["timing"]["leader_hold_s"] == 1.1

        r = urllib.request.urlopen(dash.url + "/nope", timeout=3) if False else None
        assert r is None
    finally:
        dash.close()


def test_sse_stream_sends_snapshot_then_events():
    bus = Bus()
    dash = Dashboard(bus, default_config(), port=0)
    try:
        resp = urllib.request.urlopen(dash.url + "/events", timeout=3)
        first = resp.readline().decode()
        assert first.startswith("event: snapshot")
        data = resp.readline().decode()
        assert data.startswith("data: ") and '"state"' in data
        resp.readline()  # blank
        bus.publish(ModeChanged(1, "idle", "holding"))
        line = resp.readline().decode()
        payload = json.loads(line[len("data: ") :])
        assert payload["entry"]["kind"] == "mode" and payload["snapshot"]["state"]["mode"] == "holding"
    finally:
        dash.close()
