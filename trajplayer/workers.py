from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QThread, Signal

from .ase_cache import ConversionCancelled, build_cache_from_source, open_valid_cache
from .bonds import connected_components, infer_bonds
from .frame_store import FrameStore
from .random_access_cache import (
    open_direct_random_access_store,
    supports_random_access_source,
)
from .topology import BondSource, BondTopology
from .trajectory_source import TrajectorySource


class TrajectoryOpenThread(QThread):
    preview_ready = Signal(object, object)
    loaded = Signal(object, object, bool)
    failed = Signal(str)
    progress = Signal(int, int)
    stage_changed = Signal(str)
    index_progress = Signal(int, bool)

    def __init__(self, source: TrajectorySource) -> None:
        super().__init__()
        self.source = source
        self._cancel_event = threading.Event()
        self._preview_store: FrameStore | None = None

    @property
    def preview_store(self) -> FrameStore | None:
        return self._preview_store

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        store: FrameStore | None = None
        last_progress_s = 0.0

        def emit_progress(done: int, total: int) -> None:
            nonlocal last_progress_s
            now_s = time.monotonic()
            if done == 1 or done == total or now_s - last_progress_s >= 0.05:
                last_progress_s = now_s
                self.progress.emit(done, total)

        def emit_preview(preview_store: FrameStore) -> None:
            self._preview_store = preview_store
            self.preview_ready.emit(self.source, preview_store)

        try:
            if supports_random_access_source(self.source):
                from_cache = False
                store = open_direct_random_access_store(
                    self.source,
                    status_callback=self.stage_changed.emit,
                    index_progress_callback=self.index_progress.emit,
                )
                emit_preview(store)
                known_count = store.frame_count
                self.index_progress.emit(known_count, store.frame_count_is_final)
                while not self._cancel_event.is_set() and not store.frame_count_is_final:
                    known_count, complete = store.wait_for_index_update(
                        known_count,
                        timeout_s=0.05,
                    )
                    self.index_progress.emit(known_count, complete)
            else:
                try:
                    store = open_valid_cache(self.source)
                    from_cache = True
                except Exception:
                    from_cache = False
                    store = build_cache_from_source(
                        self.source,
                        progress_callback=emit_progress,
                        preview_callback=emit_preview,
                        cancel_event=self._cancel_event,
                    )
            if self._cancel_event.is_set():
                if store is not None and self._preview_store is None:
                    store.close()
                return
            self.loaded.emit(self.source, store, from_cache)
        except ConversionCancelled:
            if store is not None and self._preview_store is None:
                store.close()
        except Exception as exc:
            failed_store = store if store is not None else self._preview_store
            if failed_store is not None and self._preview_store is None:
                failed_store.close()
            traceback.print_exc()
            self.failed.emit(f"Failed to open trajectory:\n{exc}")


class BondInferenceThread(QThread):
    ready = Signal(int, object, float)
    failed = Signal(int, str)

    def __init__(
        self,
        generation: int,
        positions: np.ndarray,
        atom_numbers: np.ndarray,
        cell: np.ndarray | None,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.generation = int(generation)
        self.positions = np.asarray(positions, dtype=np.float32)
        if not self.positions.flags.c_contiguous:
            raise ValueError("Bond inference positions must be contiguous float32 data")
        self.atom_numbers = np.ascontiguousarray(atom_numbers, dtype=np.uint16)
        self.cell = None if cell is None else np.ascontiguousarray(cell, dtype=np.float32)
        self._release_callback = release_callback

    def run(self) -> None:
        start = time.perf_counter()
        try:
            bonds = infer_bonds(
                self.positions,
                self.atom_numbers,
                cell=self.cell,
                cancelled=lambda: self.isInterruptionRequested(),
            )
            if self.isInterruptionRequested():
                return
            component_ids, component_sizes = connected_components(len(self.atom_numbers), bonds)
            if self.isInterruptionRequested():
                return
            topology = BondTopology(
                bonds=bonds,
                component_ids=component_ids,
                component_sizes=component_sizes,
                source=BondSource.INFERRED_STATIC,
                source_frame=0,
            )
            self.ready.emit(
                self.generation,
                topology,
                (time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(self.generation, f"Failed to infer bonds: {exc}")
        finally:
            release = self._release_callback
            self._release_callback = None
            if release is not None:
                release()

