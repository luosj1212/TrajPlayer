from __future__ import annotations

import atexit
import ctypes
import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path


_DLL_DIR_HANDLES: list[object] = []
_GPU_DRIVER_HANDLES: list[object] = []
_TIMER_RESOLUTION_ENABLED = False
_ERROR_LOG_PATH: Path | None = None
ERROR_LOG_NAME = "traj_player_error.log"
ERROR_LOG_MAX_BYTES = 2 * 1024 * 1024
ERROR_LOG_BACKUPS = 2


def runtime_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parents[1]


def redact_path(path: str | Path | None) -> str:
    """Keep diagnostics useful without exposing the local account name."""
    if not path:
        return ""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return "<redacted>"
    for base, label in (
        (Path.home(), "<home>"),
        (Path(tempfile.gettempdir()), "<temp>"),
    ):
        try:
            relative = resolved.relative_to(base.resolve())
        except (OSError, ValueError):
            continue
        return str(Path(label) / relative)
    return str(Path("<path>") / resolved.name)


def rotate_log(
    path: Path,
    *,
    max_bytes: int = ERROR_LOG_MAX_BYTES,
    backups: int = ERROR_LOG_BACKUPS,
) -> bool:
    """Rotate a completed log before opening a new process log."""
    try:
        if not path.exists() or path.stat().st_size < max(1, int(max_bytes)):
            return False
        for index in range(max(1, int(backups)), 1, -1):
            source = path.with_name(f"{path.name}.{index - 1}")
            target = path.with_name(f"{path.name}.{index}")
            if target.exists():
                target.unlink()
            if source.exists():
                source.replace(target)
        first_backup = path.with_name(f"{path.name}.1")
        if first_backup.exists():
            first_backup.unlink()
        path.replace(first_backup)
        return True
    except OSError:
        return False


def _open_error_log(log_dir: Path):
    log_path = log_dir / ERROR_LOG_NAME
    rotate_log(log_path)
    return open(log_path, "a", encoding="utf-8", buffering=1)


def install_error_log() -> Path:
    global _ERROR_LOG_PATH
    if _ERROR_LOG_PATH is not None:
        return _ERROR_LOG_PATH

    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        log_dir = Path.home() / "Library" / "Logs" / "TrajPlayer"
    elif getattr(sys, "frozen", False):
        log_dir = Path(sys.executable).resolve().parent
    else:
        log_dir = Path(__file__).resolve().parents[1]

    try:
        log_file = _open_error_log(log_dir)
    except OSError:
        if os.name == "nt":
            state_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        elif sys.platform == "darwin":
            state_root = Path.home() / "Library" / "Logs"
        else:
            state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        log_dir = state_root / "TrajPlayer"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = _open_error_log(log_dir)

    _ERROR_LOG_PATH = Path(log_file.name).resolve()
    sys.stdout = log_file
    sys.stderr = log_file
    print("\n[startup] TrajPlayer starting", flush=True)
    print(f"[startup] executable={redact_path(getattr(sys, 'executable', ''))}", flush=True)
    print(
        f"[startup] frozen={getattr(sys, 'frozen', False)} "
        f"meipass={redact_path(getattr(sys, '_MEIPASS', ''))}",
        flush=True,
    )

    def _log_exception(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = _log_exception
    return _ERROR_LOG_PATH


def error_log_path() -> Path | None:
    return _ERROR_LOG_PATH


def configure_dll_search_path(root: Path | None = None) -> None:
    if os.name != "nt":
        return
    base_dir = (root or runtime_root()).resolve()
    dll_dirs = (
        base_dir,
        base_dir / "numpy.libs",
        base_dir / "scipy.libs",
        base_dir / "PySide6",
        base_dir / "shiboken6",
    )
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    for dll_dir in dll_dirs:
        if not dll_dir.is_dir():
            continue
        dll_dir_text = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir_text))
            except OSError:
                pass
        if dll_dir_text not in path_parts:
            path_parts.insert(0, dll_dir_text)
    os.environ["PATH"] = os.pathsep.join(path_parts)


def missing_numpy_runtime_components(
    root: Path | None = None,
    *,
    system: str | None = None,
) -> tuple[str, ...]:
    base_dir = (root or runtime_root()).resolve()
    system_name = system or platform.system()
    if system_name == "Windows":
        extension_pattern = "_multiarray_umath*.pyd"
        library_pattern = "*.dll"
    elif system_name == "Darwin":
        extension_pattern = "_multiarray_umath*.so"
        library_pattern = None
    else:
        extension_pattern = "_multiarray_umath*.so"
        library_pattern = "*.so*"

    missing: list[str] = []
    if not any((base_dir / "numpy" / "_core").glob(extension_pattern)):
        missing.append(f"numpy/_core/{extension_pattern}")
    if library_pattern is not None and not any(
        (base_dir / "numpy.libs").glob(library_pattern)
    ):
        missing.append(f"numpy.libs/{library_pattern}")
    return tuple(missing)


def format_numpy_import_error(
    error: BaseException,
    root: Path | None = None,
    *,
    system: str | None = None,
) -> str:
    base_dir = (root or runtime_root()).resolve()
    system_name = system or platform.system()
    missing = missing_numpy_runtime_components(base_dir, system=system_name)
    if missing:
        diagnosis = (
            "The portable package is incomplete or security software removed a required file.\n\n"
            f"Missing: {', '.join(missing)}"
        )
    else:
        diagnosis = (
            f"A bundled NumPy component is present but {system_name} could not load it. "
            "Security software or an incomplete extraction is the most likely cause."
        )
    if system_name == "Windows":
        recovery = (
            "Download the Windows ZIP from GitHub Releases again, extract the complete TrajPlayer "
            "folder, and keep _internal beside TrajPlayer.exe. Do not run the EXE inside the ZIP or "
            "send the EXE by itself. Check Windows Security > Protection history if the file keeps "
            "disappearing."
        )
    elif system_name == "Darwin":
        recovery = (
            "Download the macOS ZIP for your Mac architecture from GitHub Releases again, "
            "extract TrajPlayer.app completely, and do not move files out of the app bundle."
        )
    else:
        recovery = (
            "Download the Linux tar.gz from GitHub Releases again, extract the complete TrajPlayer "
            "folder, and keep _internal beside the TrajPlayer executable."
        )
    if _ERROR_LOG_PATH is not None:
        log_path = _ERROR_LOG_PATH
    elif system_name == "Darwin":
        log_path = Path.home() / "Library" / "Logs" / "TrajPlayer" / ERROR_LOG_NAME
    else:
        log_path = Path(sys.executable).resolve().parent / ERROR_LOG_NAME
    technical_detail = " ".join(str(error).split())
    if len(technical_detail) > 300:
        technical_detail = technical_detail[:297] + "..."
    return (
        "TrajPlayer could not load its bundled NumPy runtime.\n\n"
        f"{diagnosis}\n\n"
        f"{recovery}\n\n"
        f"Technical detail: {type(error).__name__}: {technical_detail}\n"
        f"Log: {log_path}"
    )


def report_numpy_import_error(error: BaseException) -> None:
    traceback.print_exception(type(error), error, error.__traceback__)
    message = format_numpy_import_error(error)
    print(f"[startup] {message}", flush=True)
    if os.name == "nt" and os.environ.get("TRAJPLAYER_STARTUP_NO_DIALOG") != "1":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "TrajPlayer startup error", 0x10)
        except Exception:
            pass


def prefer_high_performance_gpu() -> None:
    if os.name == "nt":
        for dll_name in ("nvapi64.dll", "nvapi.dll"):
            try:
                _GPU_DRIVER_HANDLES.append(ctypes.WinDLL(dll_name))
                break
            except OSError:
                continue
        return

    if sys.platform.startswith("linux") and Path("/proc/driver/nvidia/gpus").exists():
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


def enable_high_resolution_timers() -> None:
    global _TIMER_RESOLUTION_ENABLED
    if os.name != "nt" or _TIMER_RESOLUTION_ENABLED:
        return
    try:
        result = ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        return
    if result == 0:
        _TIMER_RESOLUTION_ENABLED = True
        atexit.register(lambda: ctypes.windll.winmm.timeEndPeriod(1))


def initialize_runtime() -> None:
    install_error_log()
    configure_dll_search_path()
    prefer_high_performance_gpu()
    enable_high_resolution_timers()
