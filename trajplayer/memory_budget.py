from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024
MIN_FRAME_CACHE_BYTES = 64 * MIB
MAX_FRAME_CACHE_BYTES = 256 * MIB
FALLBACK_AVAILABLE_MEMORY_BYTES = 8 * 1024 * MIB
FRAME_CACHE_MEMORY_FRACTION = 0.025
MEMORY_BUDGET_QUANTUM_BYTES = 32 * MIB
MIN_RESIDENT_FRAMES = 4
PRESSURE_HEADROOM_BYTES = 512 * MIB
DYNAMIC_CACHE_FLOOR_BYTES = 32 * MIB
DYNAMIC_CACHE_STEP_BYTES = 32 * MIB
DYNAMIC_GROW_HOLD_S = 5.0
DYNAMIC_SHRINK_HOLD_S = 15.0


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


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    available_bytes: int
    process_rss_bytes: int
    cache_hit_rate: float
    decode_latency_ms: float
    decode_mb_s: float
    playback_fps: float
    interactive: bool


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    target_cache_bytes: int
    reason: str


class MemoryBudgetPolicy:
    """Stateful cache-byte policy with fast pressure shrink and slow hysteresis."""

    def __init__(self, *, frame_bytes: int, ceiling_bytes: int) -> None:
        self.frame_bytes = max(1, int(frame_bytes))
        self.ceiling_bytes = max(self.frame_bytes, int(ceiling_bytes))
        self._grow_candidate_since_s: float | None = None
        self._shrink_candidate_since_s: float | None = None

    def decide(
        self,
        snapshot: MemorySnapshot,
        *,
        current_bytes: int,
        now_s: float,
    ) -> BudgetDecision:
        floor = min(
            self.ceiling_bytes,
            max(DYNAMIC_CACHE_FLOOR_BYTES, self.frame_bytes * MIN_RESIDENT_FRAMES),
        )
        current = max(floor, min(self.ceiling_bytes, int(current_bytes)))
        available_target = max(
            floor,
            min(
                self.ceiling_bytes,
                int(max(0, snapshot.available_bytes) * FRAME_CACHE_MEMORY_FRACTION),
            ),
        )
        if snapshot.available_bytes < PRESSURE_HEADROOM_BYTES:
            self._grow_candidate_since_s = None
            self._shrink_candidate_since_s = None
            return BudgetDecision(
                target_cache_bytes=max(floor, min(available_target, current // 2)),
                reason="memory-pressure",
            )

        required_decode_mb_s = (
            self.frame_bytes * max(1.0, snapshot.playback_fps) / MIB
        )
        can_grow = (
            not snapshot.interactive
            and snapshot.cache_hit_rate < 0.70
            and snapshot.decode_mb_s > required_decode_mb_s * 1.2
            and current < available_target
        )
        if can_grow:
            self._shrink_candidate_since_s = None
            if self._grow_candidate_since_s is None:
                self._grow_candidate_since_s = float(now_s)
            if now_s - self._grow_candidate_since_s >= DYNAMIC_GROW_HOLD_S:
                self._grow_candidate_since_s = float(now_s)
                return BudgetDecision(
                    target_cache_bytes=min(
                        available_target,
                        current + DYNAMIC_CACHE_STEP_BYTES,
                    ),
                    reason="low-hit-rate-grow",
                )
            return BudgetDecision(current, "grow-hysteresis")

        self._grow_candidate_since_s = None
        can_shrink = snapshot.cache_hit_rate > 0.95 and current > floor
        if can_shrink:
            if self._shrink_candidate_since_s is None:
                self._shrink_candidate_since_s = float(now_s)
            if now_s - self._shrink_candidate_since_s >= DYNAMIC_SHRINK_HOLD_S:
                self._shrink_candidate_since_s = float(now_s)
                return BudgetDecision(
                    target_cache_bytes=max(floor, current - DYNAMIC_CACHE_STEP_BYTES),
                    reason="high-hit-rate-shrink",
                )
            return BudgetDecision(current, "shrink-hysteresis")

        self._shrink_candidate_since_s = None
        return BudgetDecision(current, "steady")


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
    host_available = pages * page_size
    cgroup_available = _cgroup_memory_available_bytes()
    if cgroup_available is None:
        return host_available
    return min(host_available, cgroup_available)


def _cgroup_memory_available_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    candidates = (
        (root / "memory.max", root / "memory.current"),
        (
            root / "memory" / "memory.limit_in_bytes",
            root / "memory" / "memory.usage_in_bytes",
        ),
    )
    for limit_path, usage_path in candidates:
        try:
            limit_text = limit_path.read_text(encoding="ascii").strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            usage = int(usage_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if limit <= 0 or limit >= (1 << 60):
            continue
        return max(0, limit - max(0, usage))
    return None


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
