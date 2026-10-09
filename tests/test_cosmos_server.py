"""tools/cosmos_server.py: the request shape the intent pipeline sends becomes transformers messages."""

import base64
import importlib.util
import json
import sys
import threading
import urllib.request
from pathlib import Path

spec = importlib.util.spec_from_file_location("cosmos_server", Path(__file__).parent.parent / "tools" / "cosmos_server.py")
cs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cs)


def body():
    png = base64.b64encode(b"\x89PNG fake").decode()
    return {
        "model": "x",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + png}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
                {"type": "text", "text": "fill the JSON"},
            ]},
        ],
        "media_io_kwargs": {"video": {"fps": 4}},
        "max_tokens": 64,
    }


def test_to_messages_keeps_order_and_counts_media():
    messages, counts = cs.to_messages(body(), decode_media=False)
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    assert [p["type"] for p in messages[1]["content"]] == ["video", "image", "text"]
    assert counts == {"images": 1, "videos": 1, "video_frames": 0}


def test_data_uri_and_completion_shape():
    mime, raw = cs._data_uri_bytes("data:image/jpeg;base64," + base64.b64encode(b"abc").decode())
    assert mime == "image/jpeg" and raw == b"abc"
    c = cs.completion("hi", "m", 3, 1)
    assert c["choices"][0]["message"]["content"] == "hi" and c["usage"]["total_tokens"] == 4


def test_self_test_server_answers_the_pipeline_client(monkeypatch):
    srv = cs.ThreadingHTTPServer(("127.0.0.1", 0), cs.make_handler(None, "m"))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            assert json.loads(r.read())["data"][0]["id"] == "m"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body()).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            out = json.loads(r.read())
        assert json.loads(out["choices"][0]["message"]["content"])["counts"]["images"] == 1
        # the pipeline's own client, pointed at it, gets a completion with no key set
        from visual_actions.intent.cosmos import CosmosReason

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("COSMOS_URL", f"http://127.0.0.1:{port}/v1")
        c = CosmosReason(media="frames")
        text = c._complete(body())
        assert "self_test" in text
    finally:
        srv.shutdown()
