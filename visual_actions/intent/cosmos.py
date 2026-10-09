"""Pass 1: NVIDIA Cosmos Reason watches the clip.

The judge sees frames (and a skeleton strip) of a span and answers physical questions: is a
person there, is a hand there, is the arm raised toward the camera, is the face touched, what
shape is the hand making, how is it moving, and whether the person looks like they are giving
a command. It never sees the engine's vocabulary or what the engine did, so its verdict is an
independent witness, not a rubber stamp.

Endpoint: NVIDIA's hosted OpenAI-compatible chat completions API
    POST https://integrate.api.nvidia.com/v1/chat/completions
    Authorization: Bearer $NVIDIA_API_KEY
    {"model": "nvidia/cosmos-reason2-8b", "messages": [...], "max_tokens": ..., ...}
Content parts, per the Cosmos Reason2 NIM reference
(https://docs.nvidia.com/nim/vision-language-models/latest/examples/cosmos-reason2/api.html):
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}  (+ "media_io_kwargs": {"video": {"fps": 4}})
Cosmos Reason is trained on 4 fps video. `media="auto"` sends a 4 fps mp4 packed from the frames
first and falls back to the frames as image parts when the endpoint rejects the video part.

Only the standard library talks HTTP (urllib); the opener and the clock are injectable for tests.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import clips
from .schema import (
    CosmosVerdict,
    Intent,
    Motion,
    Segment,
    SegmentKind,
    append_jsonl,
    by_id,
    read_jsonl,
)

DEFAULT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/cosmos-reason2-8b"
FPS = 4.0

SYSTEM_PROMPT = """You are a careful visual analyst reviewing short webcam clips.

Setting: a person sits at a laptop. The laptop's webcam faces them. The person sometimes controls the computer with hand gestures held up toward the camera; the rest of the time they type, read, talk, drink, touch their face or hair, scratch, stretch, or are simply not there. Your job is to describe what the person's body and hand are physically doing and to decide whether they were deliberately addressing the camera with their hand, without guessing what any software did.

Answer the question in the following format: <think>
your reasoning
</think>

<answer>
your answer
</answer>.

The answer must be one JSON object and nothing else."""

ANSWER_SCHEMA = """{
  "person_present": true | false,
  "hand_present": true | false,
  "attention_to_screen": "yes" | "partly" | "no" | "unknown",
  "arm_raised_toward_camera": true | false,
  "face_touched": true | false,
  "hand_description": "which fingers are extended or curled, palm orientation (toward camera / sideways / down), hand position in the frame and where it travels",
  "motion": "still" | "moving" | "swipe" | "slide" | "pinch_drag",
  "intent": "command" | "incidental" | "dead" | "unsure",
  "reasoning": "one or two sentences",
  "confidence": 0.0 to 1.0,
  "sub_spans": [{"t0": seconds_from_clip_start, "t1": seconds_from_clip_start, "intent": "command" | "incidental" | "dead" | "unsure"}]
}"""

GLOSSARY = """Definitions:
- motion "still": the hand holds one shape roughly in place. "moving": the hand travels but the shape is the point (for example carrying a pinch across the frame). "swipe": a stroke across the frame is itself the gesture. "slide": a held shape steps sideways once or a few times. "pinch_drag": thumb and index pinched together and dragged.
- intent "command": the hand is raised and shown to the camera on purpose, a deliberate pose or stroke addressed to the camera. "incidental": a hand is visible but busy with something else (typing, scratching, drinking, touching the face, gesturing while talking, resting). "dead": nobody there, or no hand visible. "unsure": you cannot tell.
- sub_spans: leave empty when the whole clip is one thing. When the clip contains more than one thing (for example 3 seconds of empty desk, then 3 seconds of a raised hand), list each span with its own intent. Times are seconds from the start of the clip."""


def build_prompt(segment: Segment, n_frames: int, has_strip: bool, fps: float = FPS, as_video: bool = False) -> tuple[str, str]:
    """(system prompt, user text) for a segment. Mentions only the clip's shape, never the engine."""
    dur = max(0.0, segment.t1 - segment.t0)
    if as_video:
        media = f"You are given a {dur:.1f} second clip at {fps:g} frames per second."
    else:
        media = f"You are given {n_frames} frames sampled at {fps:g} per second from a {dur:.1f} second clip, in order; frame k is at {1 / fps:.2f} * k seconds."
    strip = ""
    if has_strip:
        strip = " The last image is a strip of hand-landmark skeletons (21 points, green bones, red joints, mirrored like a selfie) from a hand tracker over the same span, one cell per time step with its timestamp; it shows the hand shape when the frames are small or blurry. Trust the frames over the skeleton when they disagree."
    user = f"{media}{strip}\n\nDescribe what the person is doing and fill in this JSON:\n{ANSWER_SCHEMA}\n\n{GLOSSARY}"
    return SYSTEM_PROMPT, user


# -- judges ----------------------------------------------------------------------------


class Judge(Protocol):
    def judge(self, segment: Segment, frames: list[bytes], strip: bytes | None) -> CosmosVerdict: ...


class SpendCapReached(RuntimeError):
    pass


@dataclass
class FakeJudge:
    """Canned verdicts for tests: `verdicts` by segment_id, else a default."""

    verdicts: dict[str, CosmosVerdict] = field(default_factory=dict)
    default_intent: Intent = Intent.UNSURE
    calls: list[tuple[str, int, bool]] = field(default_factory=list)

    def judge(self, segment: Segment, frames: list[bytes], strip: bytes | None) -> CosmosVerdict:
        self.calls.append((segment.segment_id, len(frames), strip is not None))
        if segment.segment_id in self.verdicts:
            return self.verdicts[segment.segment_id]
        return CosmosVerdict(
            segment_id=segment.segment_id, person_present=True, hand_present=bool(frames), attention_to_screen="unknown",
            arm_raised_toward_camera=False, face_touched=False, intent=self.default_intent, motion=Motion.STILL,
            hand_description="", reasoning="fake", confidence=0.0, model="fake",
        )


@dataclass
class DryRunJudge:
    """Writes the prompt and the frame count per segment under `out_dir`; calls nothing."""

    out_dir: Path
    fps: float = FPS
    calls: int = 0

    def judge(self, segment: Segment, frames: list[bytes], strip: bytes | None) -> CosmosVerdict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        system, user = build_prompt(segment, len(frames), strip is not None, self.fps)
        name = segment.segment_id.replace("/", "_")
        (self.out_dir / f"{name}.prompt.txt").write_text(f"# system\n{system}\n\n# user\n{user}\n", encoding="utf-8")
        (self.out_dir / f"{name}.json").write_text(json.dumps({"segment_id": segment.segment_id, "frames": len(frames), "frame_bytes": sum(map(len, frames)), "strip": strip is not None}), encoding="utf-8")
        self.calls += 1
        return CosmosVerdict(
            segment_id=segment.segment_id, person_present=False, hand_present=False, attention_to_screen="unknown",
            arm_raised_toward_camera=False, face_touched=False, intent=Intent.UNSURE, motion=Motion.STILL,
            hand_description="", reasoning="dry run", confidence=0.0, model="dry-run",
        )


Opener = Callable[[urllib.request.Request, float], Any]


def _urlopen(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(req, timeout=timeout)


@dataclass
class CosmosReason:
    """Cosmos Reason over NVIDIA's hosted endpoint, with retries, a spend cap and a media fallback."""

    api_key: str | None = None
    model: str | None = None
    url: str = DEFAULT_URL
    max_calls: int = 50  # per-run spend cap (attempts that reached the network count)
    media: str = "auto"  # "auto" | "video" | "frames"
    fps: float = FPS
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.7
    retries: int = 4
    backoff_s: float = 1.0
    timeout_s: float = 180.0
    opener: Opener = _urlopen
    sleep: Callable[[float], None] = time.sleep
    calls: int = 0
    last_request: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("NVIDIA_API_KEY")
        self.model = self.model or os.environ.get("COSMOS_MODEL", DEFAULT_MODEL)

    # -- request shape ---------------------------------------------------------------
    def body(self, segment: Segment, frames: list[bytes], strip: bytes | None, as_video: bool) -> dict[str, Any]:
        system, user = build_prompt(segment, len(frames), strip is not None, self.fps, as_video)
        parts: list[dict[str, Any]] = []
        extra: dict[str, Any] = {}
        if as_video:
            mp4 = clips.frames_to_mp4(frames, self.fps)
            parts.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(mp4).decode()}})
            extra["media_io_kwargs"] = {"video": {"fps": self.fps}}
        else:
            for f in frames:
                parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}})
        if strip is not None:
            parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(strip).decode()}})
        parts.append({"type": "text", "text": user})
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            **extra,
        }

    def judge(self, segment: Segment, frames: list[bytes], strip: bytes | None) -> CosmosVerdict:
        if not frames:
            raise ValueError(f"{segment.segment_id}: no frames to judge")
        as_video = self.media in ("auto", "video") and clips.ffmpeg_bin("ffmpeg") is not None
        try:
            text = self._complete(self.body(segment, frames, strip, as_video))
        except urllib.error.HTTPError as e:
            if as_video and self.media == "auto" and 400 <= e.code < 500 and e.code not in (401, 403, 429):
                self.media = "frames"  # the endpoint does not take a video part: frames from now on
                as_video = False
                text = self._complete(self.body(segment, frames, strip, False))
            else:
                raise
        return parse_verdict(text, segment, model=str(self.model), fps=self.fps)

    def _complete(self, body: dict[str, Any]) -> str:
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        self.last_request = {k: v for k, v in body.items() if k != "messages"} | {"parts": [p["type"] for p in body["messages"][1]["content"]]}
        data = json.dumps(body).encode()
        attempt = 0
        while True:
            if self.calls >= self.max_calls:
                raise SpendCapReached(f"spend cap of {self.max_calls} calls reached")
            self.calls += 1
            req = urllib.request.Request(self.url, data=data, method="POST", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"})
            try:
                with self.opener(req, self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                retry = e.code == 429 or 500 <= e.code < 600
                if not retry or attempt >= self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt >= self.retries:
                    raise
            self.sleep(self.backoff_s * (2**attempt))
            attempt += 1
        return _message_text(payload)


def _message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in response: {json.dumps(payload)[:300]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some servers return parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    text = content or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning and "<think>" not in text:
        text = f"<think>\n{reasoning}\n</think>\n{text}"
    return text


# -- parsing ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict[str, Any]:
    """The JSON object in a model answer: inside <answer>, inside a fence, or the last {...}."""
    candidates: list[str] = []
    m = _ANSWER.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.extend(_FENCE.findall(text))
    candidates.append(text)
    for c in candidates:
        for s in _objects(c):
            d = _loads_sloppy(s)
            if isinstance(d, dict):
                return d
    raise ValueError(f"no JSON object in answer: {text[:200]!r}")


def _objects(text: str) -> Iterable[str]:
    """Balanced {...} spans, last one first; a truncated trailing object is closed up."""
    spans: list[str] = []
    stack: list[str] = []
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if not stack:
                if ch == "[":
                    continue  # only objects are candidates
                start = i
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
            if not stack and start >= 0:
                spans.append(text[start : i + 1])
                start = -1
    if stack and start >= 0:  # truncated answer: close what is open
        tail = text[start:].rstrip()
        if in_str:
            tail += '"'
        tail = tail.rstrip(",")
        spans.append(tail + "".join(reversed(stack)))
    return reversed(spans)


def _loads_sloppy(s: str) -> Any:
    for fix in (lambda x: x, _repair):
        try:
            return json.loads(fix(s))
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = _TRAILING_COMMA.sub(r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"//[^\n]*", "", s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    return s


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("true", "yes", "y", "1")


def _enum(cls: Any, v: Any, default: Any) -> Any:
    try:
        return cls(str(v).strip().lower().replace("-", "_").replace(" ", "_"))
    except ValueError:
        return default


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return default


def parse_verdict(text: str, segment: Segment, model: str = "", fps: float = FPS) -> CosmosVerdict:
    """A CosmosVerdict from the model text; sub_span times become session elapsed seconds."""
    d = extract_json(text)
    spans: list[dict[str, Any]] = []
    for s in d.get("sub_spans") or []:
        if not isinstance(s, dict):
            continue
        try:
            a, b = float(s.get("t0", 0.0)), float(s.get("t1", 0.0))
        except (TypeError, ValueError):
            continue
        if a > segment.t1 - segment.t0 + 1.0 and a >= segment.t0:  # already absolute
            t0, t1 = a, b
        else:
            t0, t1 = segment.t0 + a, segment.t0 + b
        spans.append({"t0": round(t0, 3), "t1": round(t1, 3), "intent": _enum(Intent, s.get("intent"), Intent.UNSURE).value})
    att = str(d.get("attention_to_screen", "unknown")).strip().lower()
    if att not in ("yes", "partly", "no", "unknown"):
        att = "partly" if att in ("partial", "sometimes", "some") else "unknown"
    return CosmosVerdict(
        segment_id=segment.segment_id,
        person_present=_bool(d.get("person_present", False)),
        hand_present=_bool(d.get("hand_present", False)),
        attention_to_screen=att,
        arm_raised_toward_camera=_bool(d.get("arm_raised_toward_camera", False)),
        face_touched=_bool(d.get("face_touched", False)),
        intent=_enum(Intent, d.get("intent"), Intent.UNSURE),
        motion=_enum(Motion, d.get("motion"), Motion.STILL),
        hand_description=str(d.get("hand_description") or ""),
        reasoning=str(d.get("reasoning") or ""),
        confidence=_float(d.get("confidence"), 0.0),
        sub_spans=spans,
        model=model,
        raw=text,
    )


# -- the pass --------------------------------------------------------------------------


def run_cosmos(
    session_dir: Path,
    segments_path: Path,
    out_path: Path,
    judge: Judge,
    limit: int | None = None,
    kinds: Iterable[SegmentKind | str] | None = None,
    fps: float = FPS,
    with_strip: bool = True,
    log: Callable[[str], None] | None = None,
) -> list[CosmosVerdict]:
    """Judge every segment of the chosen kinds not yet in `out_path`; append verdicts as JSONL.

    Resumable: ids already present in `out_path` are skipped, so a crash or a spend cap
    halfway leaves a file the next run continues from.
    """
    session_dir, segments_path, out_path = Path(session_dir), Path(segments_path), Path(out_path)
    wanted = {SegmentKind(k) for k in kinds} if kinds else None
    done = set(by_id(read_jsonl(out_path, CosmosVerdict)))
    todo = [s for s in read_jsonl(segments_path, Segment) if (wanted is None or s.kind in wanted) and s.segment_id not in done]
    if limit is not None:
        todo = todo[:limit]
    index = clips.VideoIndex.load(session_dir) if todo else None
    out: list[CosmosVerdict] = []
    for seg in todo:
        frames = clips.frames_for(session_dir, seg.t0, seg.t1, fps=fps, index=index)
        strip: bytes | None = None
        if with_strip:
            try:
                strip = clips.skeleton_strip(session_dir, seg.t0, seg.t1)
            except Exception as e:  # noqa: BLE001 - the strip is a bonus, the frames are the evidence
                if log:
                    log(f"{seg.segment_id}: no strip ({e})")
        try:
            v = judge.judge(seg, frames, strip)
        except SpendCapReached as e:
            if log:
                log(str(e))
            break
        append_jsonl(out_path, v)
        out.append(v)
        if log:
            log(f"{seg.segment_id}: {v.intent.value} ({v.confidence:.2f}) {v.motion.value} - {v.hand_description[:60]}")
    return out
