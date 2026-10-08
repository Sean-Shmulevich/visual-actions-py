from __future__ import annotations

import subprocess

from ..base import PermissionState, PermissionStatus

SETTINGS_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
}


class MacPermissions:
    def check(self, prompt: bool) -> PermissionStatus:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

        trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt})
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        # AVAuthorizationStatus: 0 notDetermined, 1 restricted, 2 denied, 3 authorized
        camera = {0: PermissionState.UNDETERMINED, 3: PermissionState.GRANTED}.get(status, PermissionState.DENIED)
        if camera is PermissionState.UNDETERMINED and prompt:
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, lambda ok: None)
        return PermissionStatus(
            camera=camera,
            accessibility=PermissionState.GRANTED if trusted else PermissionState.DENIED,
            settings_hint="System Settings > Privacy & Security",
        )

    def open_settings(self, which: str) -> None:
        subprocess.run(["open", SETTINGS_URLS[which]], check=False)
