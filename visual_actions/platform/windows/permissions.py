from __future__ import annotations

from ..base import PermissionState, PermissionStatus


class WindowsPermissions:
    def check(self, prompt: bool) -> PermissionStatus:
        return PermissionStatus(
            camera=PermissionState.GRANTED,
            accessibility=PermissionState.NOT_APPLICABLE,
            settings_hint="Settings > Privacy & security > Camera",
        )

    def open_settings(self, which: str) -> None:
        pass
