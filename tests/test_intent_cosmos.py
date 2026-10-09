"""Pass 1: the Cosmos Reason prompt, answer parsing, the HTTP client's retries and cap, and run_cosmos."""

import io
import json
import re
import urllib.error
from pathlib import Path
from typing import Any, Self

import pytest

from visual_actions.core import recognizer
from visual_actions.intent import cosmos
from visual_actions.intent.cosmos import (
    CosmosReason,
    DryRunJudge,
    FakeJudge,
    SpendCapReached,
    build_prompt,
    extract_json,
    parse_verdict,
    run_cosmos,
)
from visual_actions.intent.schema import (
    CosmosVerdict,
    Intent,
    Motion,
    Segment,
    SegmentKind,
    read_jsonl,
    write_jsonl,
)

from .test_intent_clips import FIRST_FRAME_S, make_session

SEG = Segment("s/fire/0000", "s", SegmentKind.FIRE, 10.0, 16.0, engine_gesture="h_left", engine_action="Cmd+Tab", engine_outcome="accepted")


def engine_tokens() -> list[str]:
    """Every gesture name defined as a string constant in core/recognizer.py."""
    return sorted({v for k, v in vars(recognizer).items() if k.isupper() and isinstance(v, str)})


def test_prompt_has_no_engine_vocabulary():
    tokens = engine_tokens()
    assert "open_palm" in tokens and "fist" in tokens and "h_left" in tokens
    system, user = build_prompt(SEG, 24, True)
    text = (system + "\n" + user).lower()
    for tok in tokens:
        assert not re.search(rf"\b{re.escape(tok)}\b", text), tok
    for leak in ("h_left", "cmd+tab", "accepted", "fire", "armed", "engine"):
        assert leak not in text
    assert "6.0 second clip" in user and "24 frames" in user and "skeleton" in user
    assert "<think>" in system and "<answer>" in system
    _, user_video = build_prompt(SEG, 24, False, as_video=True)
    assert "frames per second" in user_video and "skeleton" not in user_video


def test_extract_json_from_think_answer_fences_and_sloppy_text():
    clean = '{"intent": "command", "confidence": 0.9}'
    assert extract_json(clean) == {"intent": "command", "confidence": 0.9}
    assert extract_json(f"<think>\nlooks like {{a}} thing\n</think>\n<answer>\n{clean}\n</answer>")["intent"] == "command"
    assert extract_json(f"Sure:\n```json\n{clean}\n```\nDone.")["confidence"] == 0.9
    sloppy = "<answer>{'intent': 'command', 'person_present': True, 'sub_spans': [], }</answer>"
    assert extract_json(sloppy) == {"intent": "command", "person_present": True, "sub_spans": []}
    truncated = '<answer>{"intent": "incidental", "reasoning": "typing", "sub_spans": [{"t0": 0, "t1": 2, "intent": "dead"}'
    assert extract_json(truncated)["intent"] == "incidental"
    prose = 'The person is typing. {"intent": "incidental", "confidence": 0.7} That is all.'
    assert extract_json(prose)["intent"] == "incidental"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_verdict_coerces_and_rebases_sub_spans():
    text = """<think>hand up, then gone</think>
<answer>
{
  "person_present": "yes", "hand_present": true, "attention_to_screen": "Partly",
  "arm_raised_toward_camera": 1, "face_touched": false,
  "hand_description": "index and middle extended, palm to camera",
  "motion": "Still", "intent": "COMMAND", "reasoning": "deliberate V sign", "confidence": "0.85",
  "sub_spans": [{"t0": 0, "t1": 2.5, "intent": "dead"}, {"t0": 2.5, "t1": 6, "intent": "command"}, "junk"]
}
</answer>"""
    v = parse_verdict(text, SEG, model="m")
    assert isinstance(v, CosmosVerdict) and v.segment_id == SEG.segment_id and v.model == "m" and v.raw == text
    assert v.person_present and v.hand_present and v.arm_raised_toward_camera and not v.face_touched
    assert v.attention_to_screen == "partly" and v.intent is Intent.COMMAND and v.motion is Motion.STILL
    assert v.confidence == 0.85
    assert v.sub_spans == [{"t0": 10.0, "t1": 12.5, "intent": "dead"}, {"t0": 12.5, "t1": 16.0, "intent": "command"}]
    weird = parse_verdict('{"intent": "deliberate", "motion": "pinch-drag", "confidence": 7, "attention_to_screen": "sometimes"}', SEG)
    assert weird.intent is Intent.UNSURE and weird.motion is Motion.PINCH_DRAG and weird.confidence == 1.0 and weird.attention_to_screen == "partly"


# -- the HTTP client ------------------------------------------------------------------


class Resp:
    def __init__(self, text: str) -> None:
        self._body = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *a: object) -> None:
        pass


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b""))  # type: ignore[arg-type]


ANSWER = '<answer>{"person_present": true, "hand_present": true, "attention_to_screen": "yes", "arm_raised_toward_camera": true, "face_touched": false, "hand_description": "open hand", "motion": "still", "intent": "command", "reasoning": "r", "confidence": 0.8}</answer>'


def make_client(script: list[Any], **kw: Any) -> tuple[CosmosReason, list[dict[str, Any]], list[float]]:
    requests: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def opener(req: Any, timeout: float) -> Any:
        requests.append(json.loads(req.data.decode()))
        assert req.get_header("Authorization") == "Bearer key"
        r = script.pop(0)
        if isinstance(r, Exception):
            raise r
        return Resp(r)

    c = CosmosReason(api_key="key", model="nvidia/cosmos-reason2-8b", opener=opener, sleep=sleeps.append, media="frames", **kw)
    return c, requests, sleeps


FRAMES = [b"\xff\xd8jpeg-one", b"\xff\xd8jpeg-two"]


def test_request_shape_frames_mode():
    c, requests, _ = make_client([ANSWER])
    v = c.judge(SEG, FRAMES, b"\x89PNGstrip")
    assert v.intent is Intent.COMMAND and v.model == "nvidia/cosmos-reason2-8b"
    body = requests[0]
    assert body["model"] == "nvidia/cosmos-reason2-8b" and body["stream"] is False and body["max_tokens"] == c.max_tokens
    assert body["messages"][0] == {"role": "system", "content": cosmos.SYSTEM_PROMPT}
    parts = body["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and parts[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "media_io_kwargs" not in body and c.calls == 1


def test_retries_on_429_and_5xx_with_backoff():
    c, requests, sleeps = make_client([http_error(429), http_error(503), urllib.error.URLError("reset"), ANSWER], backoff_s=0.5)
    assert c.judge(SEG, FRAMES, None).intent is Intent.COMMAND
    assert len(requests) == 4 and sleeps == [0.5, 1.0, 2.0] and c.calls == 4


def test_gives_up_after_retries_and_never_retries_4xx():
    c, _, sleeps = make_client([http_error(500)] * 3, retries=2)
    with pytest.raises(urllib.error.HTTPError):
        c.judge(SEG, FRAMES, None)
    assert sleeps == [1.0, 2.0]
    c, requests, sleeps = make_client([http_error(401), ANSWER])
    with pytest.raises(urllib.error.HTTPError):
        c.judge(SEG, FRAMES, None)
    assert len(requests) == 1 and sleeps == []


def test_spend_cap_counts_every_attempt():
    c, _, _ = make_client([ANSWER, http_error(429), ANSWER], max_calls=2)
    c.judge(SEG, FRAMES, None)
    with pytest.raises(SpendCapReached):
        c.judge(SEG, FRAMES, None)  # the second judge needs two attempts; only one is left
    assert c.calls == 2


def test_missing_key_is_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("COSMOS_MODEL", "nvidia/other")
    c = CosmosReason(opener=lambda r, t: Resp(ANSWER), media="frames")
    assert c.model == "nvidia/other"
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        c.judge(SEG, FRAMES, None)


@pytest.mark.skipif(cosmos.clips.ffmpeg_bin("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_part_first_then_frames_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from visual_actions.intent.clips import frames_for

    s = make_session(tmp_path, with_idx=True)
    frames = frames_for(s, FIRST_FRAME_S, FIRST_FRAME_S + 1.0, fps=4)
    c, requests, _ = make_client([http_error(400), ANSWER])
    c.media = "auto"
    v = c.judge(SEG, frames, None)
    assert v.intent is Intent.COMMAND
    first, second = requests
    assert [p["type"] for p in first["messages"][1]["content"]] == ["video_url", "text"]
    assert first["messages"][1]["content"][0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert first["media_io_kwargs"] == {"video": {"fps": 4.0}}
    assert [p["type"] for p in second["messages"][1]["content"]] == ["image_url"] * 4 + ["text"]
    assert c.media == "frames"  # remembered: the next call goes straight to frames



def test_reasoning_content_field_is_folded_into_the_text():
    payload = {"choices": [{"message": {"content": "<answer>" + '{"intent": "dead"}' + "</answer>", "reasoning_content": "nobody there"}}]}
    text = cosmos._message_text(payload)
    assert text.startswith("<think>\nnobody there") and parse_verdict(text, SEG).intent is Intent.DEAD


# -- run_cosmos -----------------------------------------------------------------------


def segments_file(tmp_path: Path) -> Path:
    segs = [
        Segment("s/fire/0000", "s", SegmentKind.FIRE, FIRST_FRAME_S + 0.0, FIRST_FRAME_S + 1.0),
        Segment("s/dead/0000", "s", SegmentKind.DEAD, FIRST_FRAME_S + 1.0, FIRST_FRAME_S + 2.0),
        Segment("s/hand/0000", "s", SegmentKind.HAND, FIRST_FRAME_S + 2.0, FIRST_FRAME_S + 3.0),
    ]
    p = tmp_path / "intent" / "segments.jsonl"
    write_jsonl(p, segs)
    return p


def test_run_cosmos_with_fake_judge_resumes(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    segs = segments_file(s)
    out = s / "intent" / "cosmos.jsonl"
    judge = FakeJudge(default_intent=Intent.INCIDENTAL)
    first = run_cosmos(s, segs, out, judge, kinds=["fire", "dead"])
    assert [v.segment_id for v in first] == ["s/fire/0000", "s/dead/0000"]
    assert judge.calls == [("s/fire/0000", 4, True), ("s/dead/0000", 4, True)]
    assert all(v.intent is Intent.INCIDENTAL for v in first)
    # second run: nothing new of those kinds; the remaining kind is judged once with a canned verdict
    assert run_cosmos(s, segs, out, judge, kinds=["fire", "dead"]) == []
    canned = CosmosVerdict("s/hand/0000", True, True, "no", False, True, Intent.INCIDENTAL, Motion.MOVING, "scratching chin", "face", 0.7)
    judge.verdicts["s/hand/0000"] = canned
    more = run_cosmos(s, segs, out, judge, with_strip=False)
    assert [v.segment_id for v in more] == ["s/hand/0000"] and more[0].face_touched
    assert judge.calls[-1] == ("s/hand/0000", 4, False)
    on_disk = list(read_jsonl(out, CosmosVerdict))
    assert [v.segment_id for v in on_disk] == ["s/fire/0000", "s/dead/0000", "s/hand/0000"]
    assert on_disk[2].motion is Motion.MOVING and on_disk[2].intent is Intent.INCIDENTAL
    assert run_cosmos(s, segs, out, judge) == []


def test_run_cosmos_limit_and_spend_cap(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    segs = segments_file(s)
    out = s / "intent" / "cosmos.jsonl"
    assert len(run_cosmos(s, segs, out, FakeJudge(), limit=1)) == 1

    class Capped:
        n = 0

        def judge(self, segment: Segment, frames: list[bytes], strip: bytes | None) -> CosmosVerdict:
            self.n += 1
            if self.n > 1:
                raise SpendCapReached("cap")
            return FakeJudge().judge(segment, frames, strip)

    logs: list[str] = []
    got = run_cosmos(s, segs, out, Capped(), log=logs.append)
    assert len(got) == 1 and any("cap" in m for m in logs)
    assert len(list(read_jsonl(out, CosmosVerdict))) == 2  # resumable: the capped one is not written


def test_dry_run_writes_prompts_and_counts(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    segs = segments_file(s)
    judge = DryRunJudge(s / "intent" / "cosmos-dry")
    got = run_cosmos(s, segs, s / "intent" / "cosmos-dry.jsonl", judge, kinds=["fire"])
    assert len(got) == 1 and got[0].model == "dry-run" and judge.calls == 1
    prompt = (s / "intent" / "cosmos-dry" / "s_fire_0000.prompt.txt").read_text()
    assert "# system" in prompt and "4 frames" in prompt
    meta = json.loads((s / "intent" / "cosmos-dry" / "s_fire_0000.json").read_text())
    assert meta == {"segment_id": "s/fire/0000", "frames": 4, "frame_bytes": meta["frame_bytes"], "strip": True} and meta["frame_bytes"] > 0


def test_cli_dispatch_runs_segments_clips_and_dry_cosmos(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from visual_actions.intent.__main__ import COMMANDS, main

    assert {"segments", "clips", "cosmos"} <= set(COMMANDS)
    s = make_session(tmp_path, with_idx=True)
    assert main(["segments", str(s)]) == 0
    assert (s / "intent" / "segments.jsonl").exists()
    segments_file(s)  # the synthetic log has no engine events; use a known set
    assert main(["clips", str(s), "--kinds", "fire", "--limit", "1"]) == 0
    assert (s / "intent" / "clips" / "s_fire_0000.f00.jpg").exists() and (s / "intent" / "clips" / "s_fire_0000.strip.png").exists()
    assert main(["cosmos", str(s), "--dry-run", "--kinds", "fire,dead"]) == 0
    assert len((s / "intent" / "cosmos-dry.jsonl").read_text().splitlines()) == 2
    assert "2 new verdicts" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["clips", str(s), "--kinds", "nope"])


def test_self_hosted_url_needs_no_key(monkeypatch):
    from visual_actions.intent.cosmos import DEFAULT_URL, CosmosReason

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("COSMOS_URL", "http://127.0.0.1:8000/v1")
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def opener(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        return Resp()

    c = CosmosReason(opener=opener, media="frames")
    assert not c.hosted and c.url == "http://127.0.0.1:8000/v1/chat/completions"
    c._complete({"model": "x", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
    assert seen["url"].startswith("http://127.0.0.1:8000") and seen["auth"] is None
    monkeypatch.delenv("COSMOS_URL")
    assert CosmosReason().url == DEFAULT_URL
