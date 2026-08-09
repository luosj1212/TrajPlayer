from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Mapping

from . import __display_version__
from .startup import redact_path


DIAGNOSTIC_PACKAGES = (
    "ase",
    "chemfiles",
    "numpy",
    "PySide6",
)


def collect_diagnostics(
    *,
    opengl: Mapping[str, object] | None = None,
    log_path: Path | None = None,
) -> dict[str, object]:
    packages: dict[str, str] = {}
    for package_name in DIAGNOSTIC_PACKAGES:
        packages[package_name] = _package_version(package_name)

    log_directory = (log_path.parent if log_path is not None else _default_state_directory())
    return {
        "trajplayer": __display_version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "supported": (3, 10) <= sys.version_info[:2] < (3, 13),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        },
        "runtime": {
            "frozen": bool(getattr(sys, "frozen", False)),
            "executable": redact_path(sys.executable),
            "qt_platform": os.environ.get("QT_QPA_PLATFORM", "default"),
        },
        "packages": packages,
        "opengl": dict(opengl) if opengl is not None else {
            "status": "available after a viewer OpenGL context is created"
        },
        "storage": {
            "log": redact_path(log_path),
            "log_directory": redact_path(log_directory),
            "log_directory_writable": directory_is_writable(log_directory),
            "trajectory_cache": (
                "direct random-access readers; XYZ/extXYZ use a small progressive "
                ".tpindex offset index"
            ),
        },
    }


def directory_is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".trajplayer-doctor-",
            dir=directory,
        ):
            pass
        return True
    except OSError:
        return False


def diagnostics_json(
    *,
    opengl: Mapping[str, object] | None = None,
    log_path: Path | None = None,
) -> str:
    return json.dumps(
        collect_diagnostics(opengl=opengl, log_path=log_path),
        indent=2,
        sort_keys=True,
    )


def probe_opengl() -> dict[str, object]:
    """Create an offscreen context and report the driver-backed GL identity."""
    try:
        from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext

        from .gl_view import default_surface_format

        app = QGuiApplication.instance()
        owns_app = app is None
        if app is None:
            app = QGuiApplication(["trajplayer-doctor"])
        surface_format = default_surface_format()
        surface = QOffscreenSurface()
        surface.setFormat(surface_format)
        surface.create()
        if not surface.isValid():
            raise RuntimeError("offscreen surface creation failed")
        context = QOpenGLContext()
        context.setFormat(surface_format)
        if not context.create() or not context.isValid():
            raise RuntimeError("OpenGL 3.3 context creation failed")
        if not context.makeCurrent(surface):
            raise RuntimeError("OpenGL context could not be made current")
        functions = context.extraFunctions()
        functions.initializeOpenGLFunctions()
        result: dict[str, object] = {
            "vendor": _decode_gl_string(functions.glGetString(0x1F00)),
            "renderer": _decode_gl_string(functions.glGetString(0x1F01)),
            "version": _decode_gl_string(functions.glGetString(0x1F02)),
            "context": (
                f"{context.format().majorVersion()}.{context.format().minorVersion()}"
            ),
        }
        context.doneCurrent()
        if owns_app:
            app.quit()
        return result
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _decode_gl_string(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        return bytes(value).decode("utf-8", errors="replace")
    except (TypeError, ValueError):
        return str(value)


def _default_state_directory() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "TrajPlayer"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "TrajPlayer"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "TrajPlayer"


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(package_name)
        except Exception:
            return "not installed"
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else "bundled; version unavailable"
