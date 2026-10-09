#!/usr/bin/env python3
"""A small OpenAI-compatible server for Cosmos Reason on a GPU box that has no vLLM.

Runs the model through Hugging Face transformers and answers the one request shape the
intent pipeline sends (visual_actions/intent/cosmos.py): POST /v1/chat/completions with a
system message and a user message whose parts are image_url / video_url data URIs plus
text. Nothing else from the OpenAI API is implemented.

On the GPU box (Python 3.10+, CUDA torch installed):

    pip install "transformers>=4.57" accelerate pillow opencv-python-headless
    pip install qwen-vl-utils            # optional: better video handling
    python tools/cosmos_server.py --model nvidia/Cosmos-Reason2-8B --port 8000

Then from the Mac, through the reverse tunnel: COSMOS_URL=http://127.0.0.1:8000/v1.

`--self-test` parses a sample request without loading the model, so the plumbing can be
checked on any machine. This file is standalone on purpose: copy it to the box alone.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# -- request -> model inputs (pure, testable without torch) -------------------------------------


def _data_uri_bytes(uri: str) -> tuple[str, bytes]:
    """'data:<mime>;base64,<payload>' -> (mime, bytes)."""
    if not uri.startswith("data:"):
        raise ValueError("only data: URIs are accepted")
    head, _, payload = uri.partition(",")
    mime = head[5:].split(";")[0]
    return mime, base64.b64decode(payload)


def video_frames(mp4: bytes, fps: float, max_frames: int = 64) -> list[Any]:
    """Decode an mp4 into PIL frames sampled at `fps`."""
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(mp4)
        path = f.name
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    every = max(1, int(round(src_fps / fps)))
    out: list[Any] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            out.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(out) >= max_frames:
                break
        i += 1
    cap.release()
    return out


def to_messages(body: dict[str, Any], decode_media: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """OpenAI chat body -> transformers chat messages ([{role, content: [parts]}]) and media counts.

    image_url parts become {"type": "image", "image": PIL}; a video_url part becomes
    {"type": "video", "video": [PIL frames]} sampled at media_io_kwargs.video.fps (default 4).
    With decode_media=False the media parts are kept as placeholders (for the self-test).
    """
    fps = float(body.get("media_io_kwargs", {}).get("video", {}).get("fps", 4))
    counts = {"images": 0, "videos": 0, "video_frames": 0}
    messages: list[dict[str, Any]] = []
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
            continue
        parts: list[dict[str, Any]] = []
        for p in content or []:
            t = p.get("type")
            if t == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif t == "image_url":
                counts["images"] += 1
                if decode_media:
                    from PIL import Image  # noqa: PLC0415

                    _, raw = _data_uri_bytes(p["image_url"]["url"])
                    parts.append({"type": "image", "image": Image.open(io.BytesIO(raw)).convert("RGB")})
                else:
                    parts.append({"type": "image", "image": "<image>"})
            elif t == "video_url":
                counts["videos"] += 1
                if decode_media:
                    _, raw = _data_uri_bytes(p["video_url"]["url"])
                    frames = video_frames(raw, fps)
                    counts["video_frames"] += len(frames)
                    parts.append({"type": "video", "video": frames})
                else:
                    parts.append({"type": "video", "video": "<video>"})
            else:
                raise ValueError(f"unsupported part type {t!r}")
        messages.append({"role": m["role"], "content": parts})
    return messages, counts


def completion(text: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


# -- the model --------------------------------------------------------------------------------------


class CosmosModel:
    def __init__(self, name: str, dtype: str = "bfloat16", device_map: str = "auto") -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        self.name = name
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=getattr(torch, dtype), device_map=device_map)
        self.model.eval()

    def generate(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float, top_p: float) -> tuple[str, int, int]:
        import torch  # noqa: PLC0415

        images = [p["image"] for m in messages for p in m["content"] if p["type"] == "image"]
        videos = [p["video"] for m in messages for p in m["content"] if p["type"] == "video"]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        inputs = self.processor(**kwargs).to(self.model.device)
        gen: dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature and temperature > 0:
            gen.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            gen["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        prompt_len = int(inputs["input_ids"].shape[1])
        new = out[0][prompt_len:]
        answer = self.processor.batch_decode([new], skip_special_tokens=True)[0]
        return answer, prompt_len, int(new.shape[0])


# -- HTTP ---------------------------------------------------------------------------------------------


def make_handler(model: CosmosModel | None, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            elif self.path.rstrip("/") in ("/health", ""):
                self._json(200, {"status": "ok", "model": model_name, "loaded": model is not None})
            else:
                self._json(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._json(404, {"error": {"message": f"no route {self.path}"}})
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                messages, counts = to_messages(body, decode_media=model is not None)
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": {"message": f"bad request: {exc}"}})
                return
            t0 = time.time()
            if model is None:
                text = json.dumps({"self_test": True, "counts": counts})
                ptok = ctok = 0
            else:
                try:
                    text, ptok, ctok = model.generate(
                        messages, int(body.get("max_tokens", 2048)), float(body.get("temperature", 0.0)), float(body.get("top_p", 1.0))
                    )
                except Exception as exc:  # noqa: BLE001
                    self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                    return
            print(f"{self.client_address[0]} {counts} {ctok} tokens {time.time() - t0:.1f}s", flush=True)
            self._json(200, completion(text, model_name, ptok, ctok))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-8B", help="HF id or local path")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="bind address: 127.0.0.1 behind an SSH tunnel, the box's Tailscale IP (or 0.0.0.0) on a tailnet")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--self-test", action="store_true", help="serve without loading the model; answers echo the parsed request")
    args = ap.parse_args()
    model = None if args.self_test else CosmosModel(args.model, args.dtype)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.model))
    print(f"cosmos server: {args.model} on http://{args.host}:{args.port}/v1 ({'self-test' if model is None else 'loaded'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
