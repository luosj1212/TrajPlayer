from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable

import numpy as np

from .binary_store import BinaryTrajectoryStore


DEFAULT_PREFETCH_MEMORY_BYTES = 256 * 1024 * 1024
DEFAULT_INTERACTIVE_PREFETCH_FRAMES = 5


class FrameStreamer:
    """Background frame loader with a byte-bounded, direction-aware RAM cache."""

    def __init__(
        self,
        store: BinaryTrajectoryStore,
        *,
        prefetch_radius: int = 200,
        max_memory_bytes: int = DEFAULT_PREFETCH_MEMORY_BYTES,
        interactive_prefetch_frames: int = DEFAULT_INTERACTIVE_PREFETCH_FRAMES,
        frame_ready_callback: Callable[[int], None] | None = None,
        error_callback: Callable[[BaseException], None] | None = None,
    ) -> None:
        if prefetch_radius < 0:
            raise ValueError("prefetch_radius must be non-negative")
        if max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        if interactive_prefetch_frames <= 0:
            raise ValueError("interactive_prefetch_frames must be positive")

        self.store = store
        self.prefetch_radius = int(prefetch_radius)
        self.max_memory_bytes = int(max_memory_bytes)
        self.interactive_prefetch_frames = int(interactive_prefetch_frames)
        self.frame_bytes = int(store.atom_count * 3 * np.dtype(np.float32).itemsize)
        if store.has_cells:
            self.frame_bytes += int(3 * 3 * np.dtype(np.float32).itemsize)
        radius_capacity = self.prefetch_radius * 2 + 1
        budget_capacity = max(1, self.max_memory_bytes // max(1, self.frame_bytes))
        self.capacity = min(store.frame_count, radius_capacity, budget_capacity)

        self._buffer = np.empty((self.capacity, store.atom_count, 3), dtype=np.float32)
        self._cell_buffer = (
            np.empty((self.capacity, 3, 3), dtype=np.float32) if store.has_cells else None
        )
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

    @property
    def memory_bytes(self) -> int:
        cell_bytes = 0 if self._cell_buffer is None else int(self._cell_buffer.nbytes)
        return int(self._buffer.nbytes) + cell_bytes

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

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
        target_indices = self._ordered_window_indices(
            frame_index,
            direction=direction,
            interactive=interactive,
        )
        target_set = set(target_indices)
        with self._ready:
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
        with self._lock:
            slot = self._index_to_slot.get(int(frame_index))
            if slot is None:
                return None
            return self._buffer[slot]

    def get_cell(self, frame_index: int) -> np.ndarray | None:
        if self._cell_buffer is None:
            return None
        with self._lock:
            slot = self._index_to_slot.get(int(frame_index))
            if slot is None:
                return None
            return self._cell_buffer[slot]

    def wait_for_frame(self, frame_index: int, *, timeout_s: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout_s
        with self._ready:
            while not self._stop:
                slot = self._index_to_slot.get(int(frame_index))
                if slot is not None:
                    return self._buffer[slot]
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

                np.copyto(self._buffer[slot], self.store.frame(frame_index))
                if self._cell_buffer is not None:
                    cell = self.store.cell(frame_index)
                    if cell is not None:
                        np.copyto(self._cell_buffer[slot], cell)

                notify_index: int | None = None
                with self._ready:
                    if self._stop or request_serial != self._request_serial:
                        self._remove_slot(slot)
                        break
                    self._index_to_slot[frame_index] = slot
                    self._slot_to_index[slot] = frame_index
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
        target_count = self.capacity
        if interactive:
            target_count = min(target_count, self.interactive_prefetch_frames)

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

    def _evict_outside(self, keep: set[int]) -> None:
        with self._lock:
            for frame_index in tuple(self._index_to_slot):
                if frame_index not in keep:
                    slot = self._index_to_slot.pop(frame_index)
                    self._slot_to_index.pop(slot, None)

    def _reserve_slot(self, keep: set[int]) -> int:
        for slot in range(self.capacity):
            if slot not in self._slot_to_index:
                return slot

        for evict_index, slot in tuple(self._index_to_slot.items()):
            if evict_index not in keep:
                self._index_to_slot.pop(evict_index, None)
                self._slot_to_index.pop(slot, None)
                return slot
        raise RuntimeError("No frame-cache slot is available")

    def _remove_slot(self, slot: int) -> None:
        old_index = self._slot_to_index.pop(slot, None)
        if old_index is not None:
            self._index_to_slot.pop(old_index, None)

    def _notify_frame_ready(self, frame_index: int) -> None:
        callback = self._frame_ready_callback
        if callback is not None:
            callback(int(frame_index))
