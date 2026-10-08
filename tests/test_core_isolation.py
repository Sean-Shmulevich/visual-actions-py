"""core/ must not import platform code or heavy libraries at module level."""

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "visual_actions" / "core"
FORBIDDEN_PREFIXES = (
    "visual_actions.platform",
    "visual_actions.ui",
    "visual_actions.plugins",
    "cv2",
    "mediapipe",
    "objc",
    "AppKit",
    "Quartz",
    "ApplicationServices",
    "AVFoundation",
    "rumps",
    "joblib",
    "sklearn",
)


def module_level_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in tree.body:  # top level only; lazy imports inside functions are allowed
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: ".x" -> within core, fine unless it reaches platform
                mod = node.module or ""
                names.append(f"rel:{'.' * node.level}{mod}")
            else:
                names.append(node.module or "")
    return names


def test_core_has_no_platform_or_heavy_imports():
    offenders = []
    for py in CORE.glob("*.py"):
        for name in module_level_imports(py):
            if name.startswith("rel:.."):
                offenders.append((py.name, name))
            if any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                offenders.append((py.name, name))
    assert not offenders, f"core/ leaks: {offenders}"
