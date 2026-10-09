# Visual Actions

Hand-gesture control for the desktop with Vim's grammar: hold a root gesture to arm a
namespace, make one command gesture, done. macOS first; Windows and Linux behind the
same interfaces.

| Root gesture (hold ~1 s) | Mode | Commands |
|---|---|---|
| Open palm | window | hand on its side (edge to the camera, fingers together pointing left, arm level), moved up / down: scroll the active window · H left / right: previous / next tab (slide to repeat) · one finger up, then slide it left / right: previous / next desktop (one step per slide, chainable) · pinch: drag the window under your hand |
| Peace sign | media | point up: play/pause · H left / right: next / previous track · pinch, hold still a moment, then move toward your right / left: volume up / down, one step per ~5 % of frame width; the popup shows the live system volume; release to finish |

A menu stays open after each command, so commands chain without the root gesture: the 5 s timeout restarts after every command, and repeating the same command needs the hand to change shape first. A fist closes the menu (as does the timeout); set `timing.chain_commands = false` for one command per root gesture. A fist cancels in every mode. Media commands post the system media keys, so they drive
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

## Cosmos on your own GPU

The intent passes judge recorded moments with NVIDIA Cosmos Reason. Hosted, they use
`NVIDIA_API_KEY`. With a GPU box that serves Cosmos through an OpenAI-compatible server
(vLLM or NIM on its port 8000) but cannot be reached from this Mac, let the box open the
connection instead: enable Remote Login on the Mac (System Settings → General → Sharing),
give the box an SSH key, and from the box run

```sh
ssh -N -R 127.0.0.1:8000:127.0.0.1:8000 <user>@<mac>    # or autossh -M 0 ...
```

The box's Cosmos port then appears on the Mac as `127.0.0.1:8000`, and the passes use it with

```sh
COSMOS_URL=http://127.0.0.1:8000/v1 COSMOS_MODEL=nvidia/cosmos-reason2-8b \
  uv run python -m visual_actions.intent cosmos <session> --limit 20
```

No key is sent to a self-hosted URL. Tailscale on both machines does the same job without
the tunnel (point `COSMOS_URL` at the box's tailnet address).
