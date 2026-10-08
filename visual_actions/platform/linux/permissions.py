from __future__ import annotations

from ..base import PermissionState, PermissionStatus


class LinuxPermissions:
    def check(self, prompt: bool) -> PermissionStatus:
        return PermissionStatus(
            camera=PermissionState.NOT_APPLICABLE,
            accessibility=PermissionState.NOT_APPLICABLE,
            settings_hint="membership of the video group",
        )

    def open_settings(self, which: str) -> None:
        pass
