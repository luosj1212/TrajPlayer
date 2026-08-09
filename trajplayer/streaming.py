from __future__ import annotations

import threading
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .frame_store import FrameStore
from .memory_budget import (
    BudgetDecision,
    FrameCacheBudget,
    MemoryBudgetPolicy,
    MemoryBudgetManager,
    MemorySnapshot,
    ViewerMemoryAllocation,
    available_memory_bytes,
)
from .process_memory import process_memory_snapshot
from .telemetry import RollingLatency


DEFAULT_INTERACTIVE_PREFETCH_FRAMES = 5
DEFAULT_IDLE_PREFETCH_FRAMES = 64
LOAD_LATENCY_EWMA_ALPHA = 0.2
CACHE_SLAB_TARGET_BYTES = 16 * 1024 * 1024
INITIAL_CACHE_TARGET_BYTES = 64 * 1024 * 1024
MEMORY_POLICY_SAMPLE_INTERVAL_S = 1.0


@dataclass(frozen=True)
class FrameStreamerStats:
    loads: int
    cache_hits: int
    cache_misses: int
    load_latency_ms: float
    decoded_megabytes: float
    decode_megabytes_per_second: float
    effective_prefetch_frames: int
    read_latency_ms_p50: float
    read_latency_ms_p95: float
    read_latency_ms_p99: float
    lease_acquisitions: int
    lease_releases: int
    active_leases: int
    peak_active_leases: int
    stale_lease_releases: int
    allocated_capacity: int
    max_capacity: int
    slab_count: int
    allocated_cache_bytes: int
    memory_target_bytes: int
    memory_target_reason: str

    @property
    def cache_hit_rate(self) -> float:
        attempts = self.cache_hits + self.cache_misses
        return 0.0 if attempts == 0 else self.cache_hits / attempts


class FrameLease:
    """Pins one cache slot while the consumer owns the current frame reference."""

    __slots__ = (
        "frame_index",
        "positions",
        "cell",
        "_owner",
        "_slot",
        "_epoch",
    )

    def __init__(
        self,
        owner: "FrameStreamer",
        *,
        frame_index: int,
        slot: int,
        epoch: int,
    ) -> None:
        self.frame_index = int(frame_index)
        self.positions = _readonly_view(owner._positions_for_slot(slot))
        self.cell = (
            None
            if not owner._cell_slabs
            else _readonly_view(owner._cell_for_slot(slot))
        )
        self._owner: FrameStreamer | None = owner
        self._slot = int(slot)
        self._epoch = int(epoch)

    @property
    def released(self) -> bool:
        return self._owner is None

    def release(self) -> None:
        owner = self._owner
        if owner is None:
            return
        self._owner = None
        owner._release_slot(self._slot, self._epoch)

    def __enter__(self) -> "FrameLease":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


class FrameStreamer:
    """Background frame loader with a byte-bounded, direction-aware RAM cache."""

    def __init__(
        self,
        store: FrameStore,
        *,
        prefetch_radius: int = 200,
        max_memory_bytes: int | None = None,
        interactive_prefetch_frames: int = DEFAULT_INTERACTIVE_PREFETCH_FRAMES,
        slab_target_bytes: int = CACHE_SLAB_TARGET_BYTES,
        initial_cache_bytes: int = INITIAL_CACHE_TARGET_BYTES,
        frame_ready_callback: Callable[[int], None] | None = None,
        error_callback: Callable[[BaseException], None] | None = None,
    ) -> None:
        if prefetch_radius < 0:
            raise ValueError("prefetch_radius must be non-negative")
        if max_memory_bytes is not None and max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        if interactive_prefetch_frames <= 0:
            raise ValueError("interactive_prefetch_frames must be positive")
        if slab_target_bytes <= 0 or initial_cache_bytes <= 0:
            raise ValueError("slab and initial cache byte targets must be positive")

        self.store = store
        self.prefetch_radius = int(prefetch_radius)
        self.interactive_prefetch_frames = int(interactive_prefetch_frames)
        self.frame_bytes = int(store.atom_count * 3 * np.dtype(np.float32).itemsize)
        if store.has_cells:
            self.frame_bytes += int(3 * 3 * np.dtype(np.float32).itemsize)
        self._dynamic_cache_enabled = max_memory_bytes is None
        if max_memory_bytes is None:
            self.memory_allocation = MemoryBudgetManager(
                atom_count=store.atom_count,
            ).allocate(
                frame_bytes=self.frame_bytes,
                frame_count=store.frame_count,
                prefetch_radius=self.prefetch_radius,
            )
            self.memory_budget = self.memory_allocation.frame_cache
            self.max_memory_bytes = self.memory_budget.bytes
        else:
            self.max_memory_bytes = int(max_memory_bytes)
            self.memory_budget = FrameCacheBudget(
                bytes=self.max_memory_bytes,
                available_memory_bytes=0,
                reserved_working_set_bytes=0,
                mode="fixed",
            )
            self.memory_allocation = ViewerMemoryAllocation(
                frame_cache=self.memory_budget,
                renderer_bytes=0,
                topology_bytes=0,
                persistent_writer_bytes=0,
            )
        radius_capacity = self.prefetch_radius * 2 + 1
        budget_capacity = max(1, self.max_memory_bytes // max(1, self.frame_bytes))
        source_capacity = store.frame_count if store.frame_count_is_final else radius_capacity
        self.capacity = min(source_capacity, radius_capacity, budget_capacity)
        self._slab_target_bytes = int(slab_target_bytes)
        self._initial_cache_bytes = int(initial_cache_bytes)
        self._frames_per_slab = max(
            1,
            min(self.capacity, self._slab_target_bytes // max(1, self.frame_bytes)),
        )
        self._position_slabs: list[np.ndarray] = []
        self._cell_slabs: list[np.ndarray] = []
        self._allocated_capacity = 0
        initial_capacity = min(
            self.capacity,
            max(
                self.interactive_prefetch_frames,
                self._initial_cache_bytes // max(1, self.frame_bytes),
            ),
        )
        self._ensure_allocated_capacity(initial_capacity)
        self._frame_ready_callback = frame_ready_callback
        self._error_callback = error_callback
        self._lock = threading.RLock()
        self._ready = threading.Condition(self._lock)
        self._center = 0
        self._direction = 0
        self._interactive = False
        self._generation = 0
        self._request_serial = 0
        self._target_indices: tuple[int, ...] = ()
        self._target_set: set[int] = set()
        self._stop = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._index_to_slot: dict[int, int] = {}
        self._slot_to_index: dict[int, int] = {}
        self._slot_lease_counts = np.zeros(self.capacity, dtype=np.int32)
        self._slot_epochs = np.zeros(self.capacity, dtype=np.uint64)
        self._loads = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_latency_ewma_s = 0.0
        self._decoded_bytes = 0
        self._load_time_s = 0.0
        self._effective_prefetch_frames = min(
            self.capacity,
            DEFAULT_IDLE_PREFETCH_FRAMES,
        )
        self._read_latency_ms = RollingLatency()
        self._lease_acquisitions = 0
        self._lease_releases = 0
        self._active_leases = 0
        self._peak_active_leases = 0
        self._stale_lease_releases = 0
        self._memory_policy = MemoryBudgetPolicy(
            frame_bytes=self.frame_bytes,
            ceiling_bytes=self.max_memory_bytes,
        )
        self._memory_target_bytes = (
            self.memory_bytes if self._dynamic_cache_enabled else self.max_memory_bytes
        )
        self._memory_target_capacity = (
            self._allocated_capacity if self._dynamic_cache_enabled else self.capacity
        )
        self._memory_target_reason = "initial" if self._dynamic_cache_enabled else "fixed"
        self._last_memory_policy_sample_s = 0.0

    @property
    def memory_bytes(self) -> int:
        with self._lock:
            return int(
                sum(slab.nbytes for slab in self._position_slabs)
                + sum(slab.nbytes for slab in self._cell_slabs)
            )

    @property
    def allocated_capacity(self) -> int:
        with self._lock:
            return self._allocated_capacity

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def stats_snapshot(self) -> FrameStreamerStats:
        with self._lock:
            throughput = (
                0.0
                if self._load_time_s <= 0.0
                else (self._decoded_bytes / (1024.0 * 1024.0)) / self._load_time_s
            )
            return FrameStreamerStats(
                loads=self._loads,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                load_latency_ms=self._load_latency_ewma_s * 1000.0,
                decoded_megabytes=self._decoded_bytes / (1024.0 * 1024.0),
                decode_megabytes_per_second=throughput,
                effective_prefetch_frames=self._effective_prefetch_frames,
                read_latency_ms_p50=self._read_latency_ms.percentile(50.0),
                read_latency_ms_p95=self._read_latency_ms.percentile(95.0),
                read_latency_ms_p99=self._read_latency_ms.percentile(99.0),
                lease_acquisitions=self._lease_acquisitions,
                lease_releases=self._lease_releases,
                active_leases=self._active_leases,
                peak_active_leases=self._peak_active_leases,
                stale_lease_releases=self._stale_lease_releases,
                allocated_capacity=self._allocated_capacity,
                max_capacity=self.capacity,
                slab_count=len(self._position_slabs),
                allocated_cache_bytes=self.memory_bytes,
                memory_target_bytes=self._memory_target_bytes,
                memory_target_reason=self._memory_target_reason,
            )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._error = None
            self._thread = threading.Thread(target=self._run, name="FrameStreamer", daemon=True)
            self._thread.start()

    def stop(self, *, timeout_s: float = 2.0) -> bool:
        with self._ready:
            self._stop = True
            self._ready.notify_all()
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=max(0.0, float(timeout_s)))
        stopped = not thread.is_alive()
        if stopped:
            with self._lock:
                if self._thread is thread:
                    self._thread = None
        return stopped

    def seek(self, frame_index: int, *, direction: int = 0, interactive: bool = False) -> None:
        frame_index = max(0, min(int(frame_index), self.store.frame_count - 1))
        direction = 1 if direction > 0 else -1 if direction < 0 else 0
        interactive = bool(interactive)
        self._update_memory_target(interactive=interactive)
        target_indices = self._ordered_window_indices(
            frame_index,
            direction=direction,
            interactive=interactive,
        )
        target_set = set(target_indices)
        with self._ready:
            if frame_index in self._index_to_slot:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            request_changed = (
                frame_index != self._center
                or direction != self._direction
                or interactive != self._interactive
                or target_indices != self._target_indices
            )
            self._center = frame_index
            self._direction = direction
            self._interactive = interactive
            if target_set != self._target_set:
                self._generation += 1
            self._target_indices = target_indices
            self._target_set = target_set
            if request_changed:
                self._request_serial += 1
                self._ready.notify_all()

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        warnings.warn(
            "FrameStreamer.get_frame() is unsafe for retained references; use acquire_frame()",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._lock:
            slot = self._index_to_slot.get(int(frame_index))
            if slot is None:
                return None
            return _readonly_view(self._positions_for_slot(slot))

    def has_frame(self, frame_index: int) -> bool:
        with self._lock:
            return int(frame_index) in self._index_to_slot

    def get_cell(self, frame_index: int) -> np.ndarray | None:
        warnings.warn(
            "FrameStreamer.get_cell() is unsafe for retained references; use acquire_frame()",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._cell_slabs:
            return None
        with self._lock:
            slot = self._index_to_slot.get(int(frame_index))
            if slot is None:
                return None
            return _readonly_view(self._cell_for_slot(slot))

    def acquire_frame(self, frame_index: int) -> FrameLease | None:
        with self._ready:
            index = int(frame_index)
            slot = self._index_to_slot.get(index)
            if slot is None:
                return None
            self._slot_lease_counts[slot] += 1
            self._lease_acquisitions += 1
            self._active_leases += 1
            self._peak_active_leases = max(
                self._peak_active_leases,
                self._active_leases,
            )
            return FrameLease(
                self,
                frame_index=index,
                slot=slot,
                epoch=int(self._slot_epochs[slot]),
            )

    def wait_for_lease(self, frame_index: int, *, timeout_s: float) -> FrameLease | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        index = int(frame_index)
        with self._ready:
            while not self._stop:
                slot = self._index_to_slot.get(index)
                if slot is not None:
                    self._slot_lease_counts[slot] += 1
                    self._lease_acquisitions += 1
                    self._active_leases += 1
                    self._peak_active_leases = max(
                        self._peak_active_leases,
                        self._active_leases,
                    )
                    return FrameLease(
                        self,
                        frame_index=index,
                        slot=slot,
                        epoch=int(self._slot_epochs[slot]),
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._ready.wait(timeout=remaining)
        return None

    def wait_for_frame(self, frame_index: int, *, timeout_s: float) -> np.ndarray | None:
        warnings.warn(
            "FrameStreamer.wait_for_frame() is unsafe for retained references; "
            "use wait_for_lease()",
            DeprecationWarning,
            stacklevel=2,
        )
        deadline = time.monotonic() + timeout_s
        with self._ready:
            while not self._stop:
                slot = self._index_to_slot.get(int(frame_index))
                if slot is not None:
                    return _readonly_view(self._positions_for_slot(slot))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._ready.wait(timeout=remaining)
        return None

    def cached_indices(self) -> list[int]:
        with self._lock:
            return sorted(self._index_to_slot)

    def notify_store_updated(self) -> None:
        """Wake the loader after a progressive cache publishes more frames."""
        with self._ready:
            self._target_indices = self._ordered_window_indices(
                self._center,
                direction=self._direction,
                interactive=self._interactive,
            )
            new_target_set = set(self._target_indices)
            if new_target_set != self._target_set:
                self._generation += 1
                self._target_set = new_target_set
            self._request_serial += 1
            self._ready.notify_all()

    def target_indices(
        self,
        frame_index: int,
        *,
        direction: int = 0,
        interactive: bool = False,
    ) -> tuple[int, ...]:
        direction = 1 if direction > 0 else -1 if direction < 0 else 0
        return self._ordered_window_indices(
            frame_index,
            direction=direction,
            interactive=interactive,
        )

    def target_window_count(
        self,
        frame_index: int,
        *,
        direction: int = 0,
        interactive: bool = False,
    ) -> int:
        return len(
            self.target_indices(
                frame_index,
                direction=direction,
                interactive=interactive,
            )
        )

    def is_window_ready(
        self,
        frame_index: int,
        *,
        direction: int = 0,
        interactive: bool = False,
    ) -> bool:
        target = set(
            self.target_indices(
                frame_index,
                direction=direction,
                interactive=interactive,
            )
        )
        with self._lock:
            return target.issubset(self._index_to_slot)

    def _run(self) -> None:
        try:
            self._run_loop()
        except Exception as exc:
            with self._ready:
                self._error = exc
                self._stop = True
                self._ready.notify_all()
            traceback.print_exc()
            callback = self._error_callback
            if callback is not None:
                try:
                    callback(exc)
                except Exception:
                    traceback.print_exc()

    def _run_loop(self) -> None:
        last_request_serial = -1
        while True:
            with self._ready:
                while not self._stop and self._request_serial == last_request_serial:
                    self._ready.wait()
                if self._stop:
                    return
                request_serial = self._request_serial
                target_indices = self._target_indices
                target_set = self._target_set.copy()
                last_request_serial = request_serial

            self._evict_outside(target_set)
            for frame_index in target_indices:
                with self._lock:
                    if self._stop or request_serial != self._request_serial:
                        break
                    if frame_index in self._index_to_slot:
                        continue
                    if not self.store.is_frame_available(frame_index):
                        continue
                    slot = self._reserve_slot(target_set)
                    while (
                        slot is None
                        and not self._stop
                        and request_serial == self._request_serial
                    ):
                        self._ready.wait()
                        slot = self._reserve_slot(target_set)
                    if self._stop or request_serial != self._request_serial:
                        break
                    if slot is None:
                        continue
                    self._slot_epochs[slot] += 1

                load_started_s = time.perf_counter()
                self.store.read_frame_into(
                    frame_index,
                    self._positions_for_slot(slot),
                    None if not self._cell_slabs else self._cell_for_slot(slot),
                )
                load_elapsed_s = max(0.0, time.perf_counter() - load_started_s)

                notify_index: int | None = None
                with self._ready:
                    if self._stop or request_serial != self._request_serial:
                        self._remove_slot(slot)
                        break
                    self._index_to_slot[frame_index] = slot
                    self._slot_to_index[slot] = frame_index
                    self._loads += 1
                    self._decoded_bytes += self.frame_bytes
                    self._load_time_s += load_elapsed_s
                    self._read_latency_ms.record(load_elapsed_s * 1000.0)
                    if self._load_latency_ewma_s <= 0.0:
                        self._load_latency_ewma_s = load_elapsed_s
                    else:
                        alpha = LOAD_LATENCY_EWMA_ALPHA
                        self._load_latency_ewma_s = (
                            alpha * load_elapsed_s
                            + (1.0 - alpha) * self._load_latency_ewma_s
                        )
                    if frame_index == self._center:
                        notify_index = frame_index
                    self._ready.notify_all()
                if notify_index is not None:
                    self._notify_frame_ready(notify_index)

    def _ordered_window_indices(
        self,
        center: int,
        *,
        direction: int,
        interactive: bool,
    ) -> tuple[int, ...]:
        center = max(0, min(int(center), self.store.frame_count - 1))
        target_count = self._adaptive_prefetch_count()
        if interactive:
            target_count = min(target_count, self.interactive_prefetch_frames)

        with self._lock:
            self._effective_prefetch_frames = target_count

        ordered = [center]
        if direction == 0:
            for distance in range(1, self.prefetch_radius + 1):
                before = center - distance
                after = center + distance
                if before >= 0:
                    ordered.append(before)
                if after < self.store.frame_count:
                    ordered.append(after)
                if len(ordered) >= target_count:
                    break
        else:
            for sign in (direction, -direction):
                for distance in range(1, self.prefetch_radius + 1):
                    index = center + sign * distance
                    if 0 <= index < self.store.frame_count:
                        ordered.append(index)
                    if len(ordered) >= target_count:
                        break
                if len(ordered) >= target_count:
                    break
        return tuple(ordered[:target_count])

    def _adaptive_prefetch_count(self) -> int:
        default_count = min(
            self.capacity,
            max(
                self.interactive_prefetch_frames,
                self._initial_cache_bytes // max(1, self.frame_bytes),
            ),
        )
        with self._lock:
            target_capacity = max(
                1,
                min(self.capacity, self._memory_target_capacity),
            )
            default_count = min(default_count, target_capacity)
            if target_capacity <= default_count:
                return target_capacity
            latency_s = self._load_latency_ewma_s
            attempts = self._cache_hits + self._cache_misses
            hit_rate = 0.0 if attempts == 0 else self._cache_hits / attempts

        if latency_s <= 0.0:
            return default_count
        frames_per_burst = max(8, int(round(0.100 / latency_s)))
        if hit_rate < 0.5:
            frames_per_burst = max(frames_per_burst, default_count)
        elif hit_rate > 0.9:
            frames_per_burst = min(frames_per_burst, default_count)
        return max(
            1,
            min(
                self.capacity,
                target_capacity,
                128,
                default_count * 2,
                frames_per_burst,
            ),
        )

    def _evict_outside(self, keep: set[int]) -> None:
        if not keep:
            return
        with self._lock:
            for frame_index in tuple(self._index_to_slot):
                if frame_index not in keep:
                    slot = self._index_to_slot[frame_index]
                    if self._slot_lease_counts[slot] == 0:
                        self._index_to_slot.pop(frame_index, None)
                        self._slot_to_index.pop(slot, None)
            self._shrink_unused_slabs(minimum_capacity=max(1, len(keep)))

    def _reserve_slot(self, keep: set[int]) -> int | None:
        for slot in range(self._allocated_capacity):
            if slot not in self._slot_to_index:
                return slot

        for evict_index, slot in tuple(self._index_to_slot.items()):
            if evict_index not in keep and self._slot_lease_counts[slot] == 0:
                self._index_to_slot.pop(evict_index, None)
                self._slot_to_index.pop(slot, None)
                return slot
        if self._allocated_capacity < self.capacity:
            slot = self._allocated_capacity
            self._ensure_allocated_capacity(slot + 1)
            return slot
        return None

    def _remove_slot(self, slot: int) -> None:
        old_index = self._slot_to_index.pop(slot, None)
        if old_index is not None:
            self._index_to_slot.pop(old_index, None)

    def _release_slot(self, slot: int, epoch: int) -> None:
        with self._ready:
            if int(self._slot_epochs[slot]) != int(epoch):
                self._stale_lease_releases += 1
                return
            if self._slot_lease_counts[slot] <= 0:
                self._stale_lease_releases += 1
                return
            self._slot_lease_counts[slot] -= 1
            self._lease_releases += 1
            self._active_leases = max(0, self._active_leases - 1)
            self._ready.notify_all()

    def _update_memory_target(self, *, interactive: bool) -> BudgetDecision | None:
        if not self._dynamic_cache_enabled:
            return None
        now_s = time.monotonic()
        with self._lock:
            if now_s - self._last_memory_policy_sample_s < MEMORY_POLICY_SAMPLE_INTERVAL_S:
                return None
            self._last_memory_policy_sample_s = now_s
            attempts = self._cache_hits + self._cache_misses
            hit_rate = 0.0 if attempts == 0 else self._cache_hits / attempts
            throughput = (
                0.0
                if self._load_time_s <= 0.0
                else (self._decoded_bytes / (1024.0 * 1024.0)) / self._load_time_s
            )
            current_bytes = self.memory_bytes
            latency_ms = self._load_latency_ewma_s * 1000.0
        available = available_memory_bytes()
        if available is None:
            available = self.memory_budget.available_memory_bytes
        rss = process_memory_snapshot().rss_bytes
        decision = self._memory_policy.decide(
            MemorySnapshot(
                available_bytes=max(0, int(available)),
                process_rss_bytes=rss,
                cache_hit_rate=hit_rate,
                decode_latency_ms=latency_ms,
                decode_mb_s=throughput,
                playback_fps=60.0,
                interactive=bool(interactive),
            ),
            current_bytes=current_bytes,
            now_s=now_s,
        )
        with self._lock:
            self._memory_target_bytes = int(decision.target_cache_bytes)
            self._memory_target_capacity = max(
                self.interactive_prefetch_frames,
                min(
                    self.capacity,
                    max(1, self._memory_target_bytes // max(1, self.frame_bytes)),
                ),
            )
            self._memory_target_reason = decision.reason
        return decision

    def _ensure_allocated_capacity(self, minimum_capacity: int) -> None:
        minimum = min(self.capacity, max(1, int(minimum_capacity)))
        while self._allocated_capacity < minimum:
            slab_frames = min(
                self._frames_per_slab,
                self.capacity - self._allocated_capacity,
            )
            self._position_slabs.append(
                np.empty((slab_frames, self.store.atom_count, 3), dtype=np.float32)
            )
            if self.store.has_cells:
                self._cell_slabs.append(
                    np.empty((slab_frames, 3, 3), dtype=np.float32)
                )
            self._allocated_capacity += slab_frames

    def _shrink_unused_slabs(self, *, minimum_capacity: int) -> None:
        minimum = max(
            1,
            min(
                self.capacity,
                max(self.interactive_prefetch_frames, int(minimum_capacity)),
            ),
        )
        while len(self._position_slabs) > 1:
            last_slab = self._position_slabs[-1]
            slab_frames = int(last_slab.shape[0])
            slab_start = self._allocated_capacity - slab_frames
            if slab_start < minimum:
                break
            slots = range(slab_start, self._allocated_capacity)
            if any(
                slot in self._slot_to_index or self._slot_lease_counts[slot] > 0
                for slot in slots
            ):
                break
            self._position_slabs.pop()
            if self._cell_slabs:
                self._cell_slabs.pop()
            self._allocated_capacity -= slab_frames

    def _positions_for_slot(self, slot: int) -> np.ndarray:
        slab_index, slab_slot = divmod(int(slot), self._frames_per_slab)
        return self._position_slabs[slab_index][slab_slot]

    def _cell_for_slot(self, slot: int) -> np.ndarray:
        slab_index, slab_slot = divmod(int(slot), self._frames_per_slab)
        return self._cell_slabs[slab_index][slab_slot]

    def _notify_frame_ready(self, frame_index: int) -> None:
        callback = self._frame_ready_callback
        if callback is not None:
            callback(int(frame_index))


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view
