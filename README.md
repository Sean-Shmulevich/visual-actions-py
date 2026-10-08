# Visual Actions

Hand-gesture control for the desktop with Vim's grammar: hold an open palm to arm a
namespace, make one command gesture, done. macOS first; Windows and Linux behind the
same interfaces.

- Product spec: Visual Actions PRD (Claude Doc)
- Technical design: [DESIGN.md](DESIGN.md)

## Develop

```sh
uv sync --extra macos --group dev
uv run pytest
uv run python -m visual_actions --replay tests/fixtures/h_left.jsonl
```
