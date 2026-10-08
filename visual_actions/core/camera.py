from __future__ import annotations

from typing import Any, Protocol


class CameraSource(Protocol):
    """Yields BGR frames. Implementations live in platform/."""

    def open(self) -> None: ...

    def read(self) -> tuple[int, Any] | None:
        """Return (t_ns, frame_bgr) or None if no frame is available."""
        ...

    def close(self) -> None: ...
