from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass


MIB = 1024 * 1024
MIN_FRAME_CACHE_BYTES = 64 * MIB
MAX_FRAME_CACHE_BYTES = 256 * MIB
FALLBACK_AVAILABLE_MEMORY_BYTES = 8 * 1024 * MIB
FRAME_CACHE_MEMORY_FRACTION = 0.025
MEMORY_BUDGET_QUANTUM_BYTES = 32 * MIB
MIN_RESIDENT_FRAMES = 4


@dataclass(frozen=True)
class FrameCacheBudget:
    bytes: int
    available_memory_bytes: int
    reserved_working_set_bytes: int
    mode: str


@dataclass(frozen=True)
class ViewerMemoryAllocation:
    frame_cache: FrameCacheBudget
    renderer_bytes: int
    topology_bytes: int
    persistent_writer_bytes: int

    @property
    def reserved_working_set_bytes(self) -> int:
        return self.renderer_bytes + self.topology_bytes + self.persistent_writer_bytes

    @property
    def total_bytes(self) -> int:
        return self.frame_cache.bytes + self.reserved_working_set_bytes


class MemoryBudgetManager:
    """Allocate frame, renderer, topology, and optional writer memory together."""

    def __init__(
        self,
        *,
        atom_count: int,
        available_bytes: int | None = None,
    ) -> None:
        self.atom_count = max(0, int(atom_count))
        self.available_bytes = available_bytes

    def allocate(
        self,
        *,
        frame_bytes: int,
        frame_count: int,
        prefetch_radius: int,
        persistent_writer_bytes: int = 0,
    ) -> ViewerMemoryAllocation:
        renderer_bytes = self.atom_count * 32
        topology_bytes = self.atom_count * 16
        writer_bytes = max(0, int(persistent_writer_bytes))
        reserved = renderer_bytes + topology_bytes + writer_bytes
        frame_cache = choose_frame_cache_budget(
            frame_bytes=frame_bytes,
            frame_count=frame_count,
            prefetch_radius=prefetch_radius,
            reserved_working_set_bytes=reserved,
            available_bytes=self.available_bytes,
        )
        return ViewerMemoryAllocation(
            frame_cache=frame_cache,
            renderer_bytes=renderer_bytes,
            topology_bytes=topology_bytes,
            persistent_writer_bytes=writer_bytes,
        )


def available_memory_bytes() -> int | None:
    """Return currently available physical memory without adding a dependency."""
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError):
            return None
        return None

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def estimate_viewer_working_set_bytes(atom_count: int) -> int:
    """Estimate non-frame CPU arrays used by topology and renderer staging."""
    return max(0, int(atom_count)) * 48


def choose_frame_cache_budget(
    *,
    frame_bytes: int,
    frame_count: int,
    prefetch_radius: int,
    reserved_working_set_bytes: int = 0,
    available_bytes: int | None = None,
) -> FrameCacheBudget:
    if frame_bytes <= 0:
        raise ValueError("frame_bytes must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if prefetch_radius < 0:
        raise ValueError("prefetch_radius must be non-negative")

    detected = available_memory_bytes() if available_bytes is None else int(available_bytes)
    mode = "auto"
    if detected is None or detected <= 0:
        detected = FALLBACK_AVAILABLE_MEMORY_BYTES
        mode = "auto-fallback"

    target_frames = min(int(frame_count), int(prefetch_radius) * 2 + 1)
    minimum_frames = min(target_frames, MIN_RESIDENT_FRAMES)
    minimum_for_frames = int(frame_bytes) * minimum_frames
    reserved = max(0, int(reserved_working_set_bytes))
    proportional = max(0, int(detected * FRAME_CACHE_MEMORY_FRACTION) - reserved)
    raw_budget = max(MIN_FRAME_CACHE_BYTES, minimum_for_frames, proportional)
    raw_budget = min(MAX_FRAME_CACHE_BYTES, raw_budget)
    quantized = max(
        MIN_FRAME_CACHE_BYTES,
        (raw_budget // MEMORY_BUDGET_QUANTUM_BYTES) * MEMORY_BUDGET_QUANTUM_BYTES,
    )
    return FrameCacheBudget(
        bytes=int(min(MAX_FRAME_CACHE_BYTES, quantized)),
        available_memory_bytes=int(detected),
        reserved_working_set_bytes=reserved,
        mode=mode,
    )
