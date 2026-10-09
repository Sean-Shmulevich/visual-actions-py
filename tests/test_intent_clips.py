"""Elapsed time -> video frame, with and without video.idx; skeleton strips."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from visual_actions.intent import clips
from visual_actions.intent.clips import (
    VideoIndex,
    frames_for,
    landmarks_between,
    sample_times,
    skeleton_strip,
)
from visual_actions.tools.synth import hand_frame

T0_NS = 1_000_000_000_000
SEGMENT_S = 2.0  # a short segment for the synthetic session
REAL_FPS = 20.0  # the camera's cadence; the writer still claims 30 fps
FIRST_FRAME_S = 0.5  # camera warm-up before the first written frame


def _gray(i: int) -> int:
    return 20 + i * 5  # frame i is a flat gray of this level, so a decoded frame names itself


def make_session(tmp_path: Path, with_idx: bool, frames_per_segment: tuple[int, ...] = (40, 40)) -> Path:
    """Two mp4v segments at the recorder's fixed 30 fps, frames really 1/REAL_FPS apart."""
    idx_lines: list[str] = []
    n_global = 0
    for seg, n in enumerate(frames_per_segment, start=1):
        w = cv2.VideoWriter(str(tmp_path / f"video-{seg:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))  # type: ignore[attr-defined]
        for i in range(n):
            w.write(np.full((48, 64, 3), _gray(i), dtype=np.uint8))
            t_ns = T0_NS + int((FIRST_FRAME_S + n_global / REAL_FPS) * 1e9)
            idx_lines.append(f"{seg} {i} {t_ns}\n")
            n_global += 1
        w.release()
    if with_idx:
        (tmp_path / "video.idx").write_text("".join(idx_lines))
    lost_frame = 10
    lost_t = FIRST_FRAME_S + lost_frame / REAL_FPS
    (tmp_path / "events.log").write_text(
        f"    0.000  00:00:00.000000  session started 2026-01-01 00:00:00 t0_ns={T0_NS}\n"
        f"    0.600  00:00:00.600000  hand    seen conf=0.99 wrist=(0.50,0.50) (mode idle)\n"
        f"{lost_t:9.3f}  00:00:01.000000  hand    lost [tracker] tracker reported no hand (frame {lost_frame}) (mode idle)\n"
        f"    3.900  00:00:03.900000  hand    seen after 2.90s away conf=0.99 wrist=(0.50,0.50) (mode idle)\n"
    )
    with (tmp_path / "landmarks.jsonl").open("w") as f:
        for k in range(8):  # a hand from 0.6 s to 1.3 s, 10 fps
            hf = hand_frame("open_palm", T0_NS + int((0.6 + k * 0.1) * 1e9))
            f.write(json.dumps({"t_ns": hf.t_ns, "hand": "right", "confidence": 0.9, "landmarks": [[lm.x, lm.y, lm.z] for lm in hf.landmarks]}) + "\n")
        f.write(json.dumps({"lost": T0_NS + int(1.4e9)}) + "\n")
    return tmp_path


def _decode(buf: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), flags)
    assert img is not None
    return img


def _level(jpeg: bytes) -> float:
    return float(_decode(jpeg, cv2.IMREAD_GRAYSCALE).mean())


def test_sample_times_half_open():
    assert sample_times(1.0, 2.0, 4) == [1.0, 1.25, 1.5, 1.75]
    assert sample_times(2.0, 1.0, 4) == []


def test_index_with_video_idx_is_exact(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    idx = VideoIndex.load(s)
    assert idx.exact_index
    assert idx.locate(0.4) is None  # before the first frame
    r = idx.locate(FIRST_FRAME_S + 10 / REAL_FPS)
    assert r is not None and (r.segment, r.index) == (1, 10)
    r = idx.locate(FIRST_FRAME_S + 45 / REAL_FPS + 0.01)  # 5 frames into segment 2
    assert r is not None and (r.segment, r.index) == (2, 5)
    assert idx.locate(99.0) is None


def test_index_without_video_idx_approximates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    s = make_session(tmp_path, with_idx=False)
    monkeypatch.setattr(clips, "SEGMENT_S", SEGMENT_S)
    idx = VideoIndex.load(s)
    assert not idx.exact_index
    start, span, n = idx.approx[1]
    assert n == 40 and span == SEGMENT_S
    assert start == pytest.approx(FIRST_FRAME_S, abs=0.05)  # from the "hand lost (frame 10)" anchor
    r = idx.locate(FIRST_FRAME_S + 10 / REAL_FPS)
    assert r is not None and r.segment == 1 and abs(r.index - 10) <= 1
    r = idx.locate(FIRST_FRAME_S + SEGMENT_S + 5 / REAL_FPS)
    assert r is not None and r.segment == 2 and abs(r.index - 5) <= 1


def test_index_without_anchor_starts_at_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    s = make_session(tmp_path, with_idx=False)
    monkeypatch.setattr(clips, "SEGMENT_S", SEGMENT_S)
    log = s / "events.log"
    log.write_text("\n".join(line for line in log.read_text().splitlines() if "lost" not in line) + "\n")
    assert VideoIndex.load(s).approx[1][0] == 0.0


@pytest.mark.parametrize("with_idx", [True, False])
def test_frames_for_returns_the_right_frames(tmp_path: Path, with_idx: bool, monkeypatch: pytest.MonkeyPatch):
    s = make_session(tmp_path, with_idx=with_idx)
    monkeypatch.setattr(clips, "SEGMENT_S", SEGMENT_S)
    t0 = FIRST_FRAME_S + 1.5  # 1.5 s into segment 1 (frame 30) through 0.5 s into segment 2 (frame 10)
    frames = frames_for(s, t0, t0 + 1.0, fps=4, width=640)
    assert len(frames) == 4
    levels = [_level(f) for f in frames]
    expected = [_gray(30), _gray(35), _gray(0), _gray(5)]  # 20 fps real: 5 frames per quarter second
    for got, want in zip(levels, expected, strict=True):
        assert abs(got - want) <= 8, (levels, expected)  # mp4v is lossy; one frame off would be 5 levels, so allow for codec noise only
    img = _decode(frames[0])
    assert img.shape[1] == 64  # never upscaled


def test_frames_for_downscales(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    img = _decode(frames_for(s, 0.6, 0.9, width=32)[0])
    assert img.shape[:2] == (24, 32)


def test_landmarks_between_uses_the_events_clock(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    got = landmarks_between(s, 0.75, 1.05)
    assert [round(t, 2) for t, _ in got] == [0.8, 0.9, 1.0]
    assert len(got[0][1]) == 21


def test_skeleton_strip_renders_png_of_expected_size(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    png = skeleton_strip(s, 0.5, 2.0, cols=8)  # the hand is tracked 0.6-1.3 s, lost after
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = _decode(png)
    assert img.shape == (120, 8 * 160, 3)
    green = (img[:, :, 1] > 200) & (img[:, :, 0] < 50)
    assert green.any(), "bones drawn where the hand was"
    assert green[:, 160:320].any()  # cell 1 is centred on 0.78 s: hand tracked
    assert not green[:, 7 * 160 :].any()  # cell 7 is centred on 1.91 s: hand lost, nothing drawn
    png_small = skeleton_strip(s, 0.5, 1.5, cols=4, cell=(80, 60))
    assert _decode(png_small).shape == (60, 320, 3)


def test_skeleton_strip_without_landmarks_is_blank(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    (s / "landmarks.jsonl").unlink()
    img = _decode(skeleton_strip(s, 0.5, 1.5, cols=2))
    assert img.shape == (120, 320, 3) and not ((img[:, :, 1] > 200) & (img[:, :, 0] < 50)).any()


@pytest.mark.skipif(clips.ffmpeg_bin("ffmpeg") is None, reason="ffmpeg not installed")
def test_clip_mp4_cuts_across_a_segment_boundary(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    out = clips.clip_mp4(s, FIRST_FRAME_S + 1.5, FIRST_FRAME_S + 2.5, tmp_path / "out" / "clip.mp4")
    cap = cv2.VideoCapture(str(out))
    n = 0
    first = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if first is None:
            first = float(img.mean())
        n += 1
    cap.release()
    assert 18 <= n <= 22  # 20 real frames in one second, written at 30 fps and cut by frame
    assert first is not None and abs(first - _gray(30)) <= 8


@pytest.mark.skipif(clips.ffmpeg_bin("ffmpeg") is None, reason="ffmpeg not installed")
def test_frames_to_mp4_packs_jpegs(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    frames = frames_for(s, 0.5, 1.5, fps=4)
    mp4 = clips.frames_to_mp4(frames, fps=4)
    assert mp4[4:8] == b"ftyp"
    p = tmp_path / "packed.mp4"
    p.write_bytes(mp4)
    cap = cv2.VideoCapture(str(p))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    cap.release()
