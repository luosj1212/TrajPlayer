from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ProcessMemorySnapshot:
    rss_bytes: int
    peak_rss_bytes: int

    @property
    def rss_mib(self) -> float:
        return self.rss_bytes / (1024.0 * 1024.0)

    @property
    def peak_rss_mib(self) -> float:
        return self.peak_rss_bytes / (1024.0 * 1024.0)


def process_memory_snapshot() -> ProcessMemorySnapshot:
    if sys.platform == "win32":
        snapshot = _windows_memory_snapshot()
    elif sys.platform == "darwin":
        snapshot = _macos_memory_snapshot()
    else:
        snapshot = _proc_memory_snapshot()
    if snapshot is not None:
        return snapshot
    peak = _resource_peak_rss_bytes()
    return ProcessMemorySnapshot(rss_bytes=peak, peak_rss_bytes=peak)


def _windows_memory_snapshot() -> ProcessMemorySnapshot | None:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            return None
        return ProcessMemorySnapshot(
            rss_bytes=int(counters.WorkingSetSize),
            peak_rss_bytes=int(counters.PeakWorkingSetSize),
        )
    except (AttributeError, OSError):
        return None


def _proc_memory_snapshot() -> ProcessMemorySnapshot | None:
    status_path = Path("/proc/self/status")
    try:
        fields: dict[str, int] = {}
        for line in status_path.read_text(encoding="ascii").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                fields[name] = int(value.strip().split()[0]) * 1024
        rss = fields.get("VmRSS")
        if rss is None:
            return None
        return ProcessMemorySnapshot(
            rss_bytes=rss,
            peak_rss_bytes=max(rss, fields.get("VmHWM", rss)),
        )
    except (OSError, ValueError, IndexError):
        return None


def _macos_memory_snapshot() -> ProcessMemorySnapshot | None:
    class TimeValue(ctypes.Structure):
        _fields_ = [("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32)]

    class MachTaskBasicInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", TimeValue),
            ("system_time", TimeValue),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        library.mach_task_self.restype = ctypes.c_uint32
        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        result = library.task_info(
            library.mach_task_self(),
            20,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result != 0:
            return None
        peak = max(int(info.resident_size), int(info.resident_size_max))
        return ProcessMemorySnapshot(rss_bytes=int(info.resident_size), peak_rss_bytes=peak)
    except (AttributeError, OSError):
        return None


def _resource_peak_rss_bytes() -> int:
    if resource is None:
        return 0
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError):
        return 0
    return value if sys.platform == "darwin" else value * 1024
