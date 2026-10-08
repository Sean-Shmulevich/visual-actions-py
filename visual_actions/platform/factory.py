"""The only `if platform == ...` in the codebase."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..core.automation import DesktopAutomation
from ..core.camera import CameraSource
from ..core.config import CameraConfig
from .base import Permissions


@dataclass
class PlatformServices:
    name: str
    camera: CameraSource
    automation: DesktopAutomation
    permissions: Permissions


def create(camera_cfg: CameraConfig, dry_run: bool = False) -> PlatformServices:
    name = current_platform()
    if name == "macos":
        from .macos.automation import MacAutomation
        from .macos.camera import OpenCVCamera
        from .macos.permissions import MacPermissions

        automation: DesktopAutomation = MacAutomation()
        camera: CameraSource = OpenCVCamera(camera_cfg)
        permissions: Permissions = MacPermissions()
    elif name == "windows":
        from .macos.camera import OpenCVCamera  # OpenCV wraps Media Foundation too
        from .windows.automation import WindowsAutomation
        from .windows.permissions import WindowsPermissions

        automation = WindowsAutomation()
        camera = OpenCVCamera(camera_cfg)
        permissions = WindowsPermissions()
    else:
        from .linux.automation import LinuxAutomation
        from .linux.permissions import LinuxPermissions
        from .mock.camera import SyntheticCamera

        automation = LinuxAutomation()
        camera = SyntheticCamera()
        permissions = LinuxPermissions()
    if dry_run:
        from .mock.automation import MockAutomation

        automation = MockAutomation(echo=True)
    return PlatformServices(name=name, camera=camera, automation=automation, permissions=permissions)


def current_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"
