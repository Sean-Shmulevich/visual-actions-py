from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PermissionState(Enum):
    GRANTED = "granted"
    DENIED = "denied"
    UNDETERMINED = "undetermined"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class PermissionStatus:
    camera: PermissionState
    accessibility: PermissionState
    settings_hint: str = ""


class Permissions(Protocol):
    def check(self, prompt: bool) -> PermissionStatus: ...

    def open_settings(self, which: str) -> None: ...
