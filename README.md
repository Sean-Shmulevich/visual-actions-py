# Visual Actions

Hand-gesture control for the desktop with Vim's grammar: hold a root gesture to arm a
namespace, make one command gesture, done. macOS first; Windows and Linux behind the
same interfaces.

| Root gesture (hold ~1 s) | Mode | Commands |
|---|---|---|
| Open palm | window | H left/right: Cmd+Tab / Cmd+Shift+Tab · point up / peace: previous / next tab (slide to repeat) · pinch: drag the window under your hand |
| Peace sign | media | point up: play/pause · thumbs up / down: next / previous track · pinch, hold still a moment, then move toward your right / left: volume up / down, one step per ~5 % of frame width; the popup shows the live system volume; release to finish |

A fist cancels in every mode. Media commands post the system media keys, so they drive
whatever app owns Now Playing (Spotify, a browser, Music).

- Product spec: Visual Actions PRD (Claude Doc)
- Technical design: [DESIGN.md](DESIGN.md)

## Develop

```sh
uv sync --extra macos --group dev
uv run pytest
uv run python -m visual_actions --replay tests/fixtures/h_left.jsonl
```

Models (not in git): `models/hand_landmarker.task` (MediaPipe hand landmarker) and
`models/blaze_face_short_range.tflite` (MediaPipe face detector, used by the face-touch veto so a
hand resting on your face is not read as the open-palm leader). Download both from
`https://storage.googleapis.com/mediapipe-models/`; without the face model the veto is off.

Every start/stop records a session under `~/Library/Application Support/visual-actions/sessions/<stamp>/`:
`video.mp4` (camera, elapsed time and mode burned in), `events.log` (timestamped modes, tokens, actions
with results, drags and snaps), and `landmarks.jsonl` (raw hand frames, same format as the datasets, so
every session is also training data). Menu bar → Open sessions folder. Disable with `record_sessions = false`.

While the app runs, a live dashboard is at http://127.0.0.1:8765 (menu bar → Open dashboard):
mode, last token and confidence, camera and gate stats, timing, drag settings, bindings, and
a rolling event log. Disable with `dashboard = false` under `[feedback]` in the config.
