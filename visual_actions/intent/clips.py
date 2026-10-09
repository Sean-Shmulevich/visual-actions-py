"""Frames, clips and skeleton strips for a span of a session.

Time runs on the events.log clock (elapsed seconds). Video lives in `video-NNN.mp4` segments
written by cv2 at a fixed 30 fps while the camera really ran at ~23 fps, so video time and
elapsed time drift apart. Two ways to map elapsed -> (segment, frame):

* `video.idx` (newer sessions): one line per written frame, "segment frame_index t_ns", joined
  to elapsed time through the session's t0_ns. Exact.
* without it: segment k starts at `first_frame_elapsed + (k - 1) * segment_s` and holds
  nb_frames frames spread evenly over its real span, so
  `frame = (elapsed - start_k) * nb_frames_k / span_k`. The first frame's elapsed time is
  estimated from the tracker's frame counter in the `hand lost (frame N)` lines (the camera
  warms up for a couple of seconds before the first frame); without those it is 0.

Everything here is offline and platform-free: cv2 for decoding and drawing, ffmpeg/ffprobe
when found on PATH (or FFMPEG / FFPROBE in the environment).
"""

from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .events import Event, parse_events, session_t0_ns

SEGMENT_S = 120.0
WRITER_FPS = 30.0
_VIDEO = re.compile(r"^video-(\d{3})\.mp4$")
_LOST_FRAME = re.compile(r"\(frame (\d+)\)")

# MediaPipe hand topology, same list as ui/preview.py (copied: intent/ never imports ui/).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ffmpeg_bin(name: str = "ffmpeg") -> str | None:
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    found = shutil.which(name)
    if found:
        return found
    brew = Path("/opt/homebrew/bin") / name
    return str(brew) if brew.exists() else None


@lru_cache(maxsize=512)
def nb_frames(path: str) -> int:
    """Frames in a video file: ffprobe when present, else cv2's count."""
    probe = ffmpeg_bin("ffprobe")
    if probe:
        try:
            out = subprocess.run(
                [probe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=60, check=False,
            ).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        except (OSError, subprocess.SubprocessError):
            pass
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def video_segments(session_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in session_dir.iterdir():
        m = _VIDEO.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


@dataclass
class FrameRef:
    segment: int
    index: int
    elapsed: float


@dataclass
class VideoIndex:
    """elapsed seconds -> (segment, frame index) for one session."""

    session_dir: Path
    segments: dict[int, Path]
    # exact: per segment, the elapsed time of each written frame (from video.idx)
    exact: dict[int, list[float]] = field(default_factory=dict)
    # approximate: per segment (start elapsed, span seconds, frame count)
    approx: dict[int, tuple[float, float, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, session_dir: Path, events: list[Event] | None = None, segment_s: float | None = None) -> VideoIndex:
        session_dir = Path(session_dir)
        segs = video_segments(session_dir)
        idx = cls(session_dir, segs)
        if events is None:
            log = session_dir / "events.log"
            events = parse_events(log) if log.exists() else []
        t0_ns = session_t0_ns(events, session_dir / "landmarks.jsonl")
        idx_path = session_dir / "video.idx"
        if idx_path.exists() and t0_ns is not None:
            with idx_path.open(encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                    seg, i, t_ns = int(parts[0]), int(parts[1]), int(parts[2])
                    times = idx.exact.setdefault(seg, [])
                    if i == len(times):  # keep the list dense and in order
                        times.append((t_ns - t0_ns) / 1e9)
            if idx.exact and all(len(v) > 0 for v in idx.exact.values()):
                return idx
            idx.exact = {}
        idx._build_approx(events, segment_s if segment_s is not None else SEGMENT_S)
        return idx

    # -- approximation --------------------------------------------------------------
    def _build_approx(self, events: list[Event], segment_s: float) -> None:
        if not self.segments:
            return
        counts = {k: nb_frames(str(p)) for k, p in self.segments.items()}
        full = [n for k, n in counts.items() if k != max(counts)] or list(counts.values())
        full_n = sorted(full)[len(full) // 2]
        fps = full_n / segment_s if full_n else WRITER_FPS
        offset = _first_frame_offset(events, fps)
        last = max(counts)
        for k in sorted(self.segments):
            start = offset + (k - 1) * segment_s
            span = segment_s if k != last else counts[k] / fps  # only the last segment is short
            self.approx[k] = (start, max(span, 1e-6), counts[k])

    # -- queries ----------------------------------------------------------------------
    @property
    def exact_index(self) -> bool:
        return bool(self.exact)

    def locate(self, elapsed: float) -> FrameRef | None:
        """The frame shown at `elapsed`, or None when it falls outside the recording."""
        if self.exact:
            for seg in sorted(self.exact):
                times = self.exact[seg]
                nxt = self.exact.get(seg + 1)
                end = nxt[0] if nxt else times[-1] + 1.0 / WRITER_FPS
                if times[0] <= elapsed < end:
                    i = max(0, bisect.bisect_right(times, elapsed) - 1)
                    return FrameRef(seg, i, times[i])
            return None
        for seg in sorted(self.approx):
            start, span, n = self.approx[seg]
            if n <= 0:
                continue
            if start <= elapsed < start + span or (seg == max(self.approx) and start <= elapsed):
                i = int((elapsed - start) * n / span)
                if i >= n:
                    return None
                return FrameRef(seg, i, start + i * span / n)
        return None

    def elapsed_of(self, segment: int, index: int) -> float | None:
        if self.exact:
            times = self.exact.get(segment)
            return times[index] if times and 0 <= index < len(times) else None
        if segment in self.approx:
            start, span, n = self.approx[segment]
            return start + index * span / n if n else None
        return None

    def video_time(self, ref: FrameRef) -> float:
        """Seconds into the segment file as a player (ffmpeg) sees it."""
        return ref.index / WRITER_FPS


def _first_frame_offset(events: list[Event], fps: float) -> float:
    """Elapsed time of the first written frame, from the tracker's frame counter when logged."""
    for e in events:
        if e.kind == "hand" and e.hand_seen is False:
            m = _LOST_FRAME.search(e.text)
            if m and fps > 0:
                return max(0.0, e.t - int(m.group(1)) / fps)
    return 0.0


def sample_times(t0: float, t1: float, fps: float) -> list[float]:
    if t1 <= t0 or fps <= 0:
        return []
    step = 1.0 / fps
    out: list[float] = []
    t = t0
    while t < t1 - 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def _encode_jpeg(img: Any, width: int, quality: int = 85) -> bytes:
    import cv2

    h, w = img.shape[:2]
    if width and w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return bytes(buf)


def frames_for(session_dir: Path, t0: float, t1: float, fps: float = 4, width: int = 640, index: VideoIndex | None = None) -> list[bytes]:
    """JPEG frames sampled at `fps` over elapsed [t0, t1), downscaled to `width` pixels wide."""
    import cv2

    session_dir = Path(session_dir)
    index = index or VideoIndex.load(session_dir)
    refs = [(t, index.locate(t)) for t in sample_times(t0, t1, fps)]
    wanted: dict[int, list[tuple[int, int]]] = {}  # segment -> [(position in output, frame index)]
    for pos, (_, ref) in enumerate(refs):
        if ref is not None:
            wanted.setdefault(ref.segment, []).append((pos, ref.index))
    out: dict[int, bytes] = {}
    for seg, items in wanted.items():
        cap = cv2.VideoCapture(str(index.segments[seg]))
        try:
            last = -2
            for pos, i in sorted(items, key=lambda x: x[1]):
                if i != last + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, img = cap.read()
                last = i
                if ok:
                    out[pos] = _encode_jpeg(img, width)
        finally:
            cap.release()
    return [out[k] for k in sorted(out)]


def clip_mp4(session_dir: Path, t0: float, t1: float, out_path: Path, index: VideoIndex | None = None, reencode: bool = True) -> Path:
    """Cut elapsed [t0, t1) into one mp4 at `out_path` (re-encoded with libx264 for frame accuracy).

    A span across a segment boundary is cut in two parts and joined with the concat demuxer.
    """
    session_dir, out_path = Path(session_dir), Path(out_path)
    index = index or VideoIndex.load(session_dir)
    ff = ffmpeg_bin("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg not found (set FFMPEG or install it)")
    a, b = index.locate(t0), index.locate(max(t0, t1 - 1e-3))
    if a is None and b is None:
        raise ValueError(f"span {t0}-{t1}s is outside the recording")
    if a is None:
        a = FrameRef(b.segment, 0, 0.0)  # type: ignore[union-attr]
    if b is None:
        last_seg = max(index.segments)
        b = FrameRef(last_seg, nb_frames(str(index.segments[last_seg])) - 1, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an"] if reencode else ["-c", "copy", "-an"]
    parts: list[tuple[int, int, int]] = []  # (segment, first frame, last frame)
    if a.segment == b.segment:
        parts.append((a.segment, a.index, b.index))
    else:
        parts.append((a.segment, a.index, nb_frames(str(index.segments[a.segment])) - 1))
        for seg in range(a.segment + 1, b.segment):
            parts.append((seg, 0, nb_frames(str(index.segments[seg])) - 1))
        parts.append((b.segment, 0, b.index))

    def cut(seg: int, i0: int, i1: int, dst: Path) -> None:
        start = i0 / WRITER_FPS
        dur = max(1, i1 - i0 + 1) / WRITER_FPS
        cmd = [ff, "-v", "error", "-y", "-ss", f"{start:.4f}", "-i", str(index.segments[seg]), "-t", f"{dur:.4f}", *codec, str(dst)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

    if len(parts) == 1:
        cut(*parts[0], out_path)
        return out_path
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for n, (seg, i0, i1) in enumerate(parts):
            p = Path(tmp) / f"part{n}.mp4"
            cut(seg, i0, i1, p)
            files.append(p)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in files), encoding="utf-8")
        subprocess.run([ff, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_path)], check=True, capture_output=True, timeout=300)
    return out_path


def frames_to_mp4(frames: list[bytes], fps: float = 4) -> bytes:
    """Pack JPEG frames into a short h264 mp4 (for judges that take a video part)."""
    ff = ffmpeg_bin("ffmpeg")
    if ff is None or not frames:
        raise RuntimeError("ffmpeg not found")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        cmd = [ff, "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, input=b"".join(frames), check=True, capture_output=True, timeout=300)
        return out.read_bytes()


# -- landmarks ------------------------------------------------------------------------

_T_NS = re.compile(r'"t_ns":\s*(\d+)')


def landmarks_between(session_dir: Path, t0: float, t1: float, t0_ns: int | None = None) -> list[tuple[float, list[list[float]]]]:
    """(elapsed, 21 x [x, y, z]) hand frames inside [t0, t1), raw camera coordinates."""
    session_dir = Path(session_dir)
    path = session_dir / "landmarks.jsonl"
    if not path.exists():
        return []
    if t0_ns is None:
        log = session_dir / "events.log"
        t0_ns = session_t0_ns(parse_events(log) if log.exists() else [], path)
    if t0_ns is None:
        return []
    lo, hi = t0_ns + int(t0 * 1e9), t0_ns + int(t1 * 1e9)
    out: list[tuple[float, list[list[float]]]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = _T_NS.match(line, 1) or _T_NS.search(line[:40])
            if not m:
                continue
            t_ns = int(m.group(1))
            if t_ns < lo:
                continue
            if t_ns >= hi:
                break
            d = json.loads(line)
            if "landmarks" in d:
                out.append(((t_ns - t0_ns) / 1e9, d["landmarks"]))
    return out


def skeleton_strip(session_dir: Path, t0: float, t1: float, cols: int = 8, cell: tuple[int, int] = (160, 120), tolerance_s: float = 0.25) -> bytes:
    """A PNG strip of `cols` cells, each the mirrored (selfie view) skeleton nearest its sample time."""
    import cv2
    import numpy as np

    w, h = cell
    img = np.full((h, w * cols, 3), 24, dtype=np.uint8)
    frames = landmarks_between(session_dir, t0 - tolerance_s, t1 + tolerance_s)
    times = [t for t, _ in frames]
    span = max(t1 - t0, 1e-6)
    for c in range(cols):
        t = t0 + span * (c + 0.5) / cols
        x0 = c * w
        cv2.rectangle(img, (x0, 0), (x0 + w - 1, h - 1), (60, 60, 60), 1)
        cv2.putText(img, f"{t:.2f}s", (x0 + 4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        if not times:
            continue
        k = bisect.bisect_left(times, t)
        best = min((j for j in (k - 1, k) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t), default=None)
        if best is None or abs(times[best] - t) > tolerance_s:
            cv2.putText(img, "no hand", (x0 + 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
            continue
        pts = [(x0 + int((1 - lm[0]) * w), int(lm[1] * h)) for lm in frames[best][1]]
        for a, b in CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), 1)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 3 if i in (4, 8, 12, 16, 20) else 2, (0, 0, 255), -1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return bytes(buf)
