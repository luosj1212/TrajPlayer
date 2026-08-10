from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .binary_store import BinaryTrajectoryStore, cache_dir_for_source, prepare_cache_directory
from .random_access_cache import open_direct_random_access_store
from .trajectory_source import TrajectorySource


ProgressCallback = Callable[[int, int], None]
PreviewCallback = Callable[[BinaryTrajectoryStore], None]


class ConversionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class AseTrajectorySummary:
    frame_count: int
    atom_count: int
    atom_numbers: np.ndarray
    symbols: list[str]
    has_cell: bool


def inspect_ase_source(source_path: Path) -> AseTrajectorySummary:
    source_path = source_path.resolve()
    direct = open_direct_random_access_store(TrajectorySource(source_path))
    try:
        frame_count = direct.frame_count
        while not direct.frame_count_is_final:
            frame_count, _complete = direct.wait_for_index_update(
                frame_count,
                timeout_s=0.05,
            )
        return AseTrajectorySummary(
            frame_count=direct.frame_count,
            atom_count=direct.atom_count,
            atom_numbers=direct.atom_numbers.copy(),
            symbols=list(direct.reader.summary.symbols),
            has_cell=direct.has_cells,
        )
    finally:
        direct.close()


def open_valid_cache(source_value: Path | TrajectorySource) -> BinaryTrajectoryStore:
    source = _coerce_source(source_value)
    root = cache_dir_for_source(source.trajectory_path)
    store = BinaryTrajectoryStore.open(root)
    if not store.is_complete or not store.is_valid_for_sources(source.paths):
        store.close()
        raise FileNotFoundError(f"Cache is missing or stale for {source.display_name}")
    return store


def build_cache_from_source(
    source: TrajectorySource,
    *,
    cache_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> BinaryTrajectoryStore:
    if source.is_gromacs_trajectory:
        from .gromacs_cache import build_cache_from_gromacs

        return build_cache_from_gromacs(
            source,
            cache_root=cache_root,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            cancel_event=cancel_event,
        )
    return build_cache_from_ase(
        source.trajectory_path,
        cache_root=cache_root,
        progress_callback=progress_callback,
        preview_callback=preview_callback,
        cancel_event=cancel_event,
    )


def build_cache_from_ase(
    source_path: Path,
    *,
    cache_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> BinaryTrajectoryStore:
    source_path = source_path.resolve()
    direct = open_direct_random_access_store(TrajectorySource(source_path))
    store: BinaryTrajectoryStore | None = None
    try:
        frame_count = direct.frame_count
        while not direct.frame_count_is_final:
            frame_count, _complete = direct.wait_for_index_update(
                frame_count,
                timeout_s=0.05,
            )
        summary = AseTrajectorySummary(
            frame_count=direct.frame_count,
            atom_count=direct.atom_count,
            atom_numbers=direct.atom_numbers.copy(),
            symbols=list(direct.reader.summary.symbols),
            has_cell=direct.has_cells,
        )
        root = cache_root.resolve() if cache_root is not None else cache_dir_for_source(source_path)
        root, temporary_cache = prepare_cache_directory(root)
        stat = source_path.stat()
        store = BinaryTrajectoryStore.create(
            root,
            frame_count=summary.frame_count,
            atom_numbers=summary.atom_numbers,
            symbols=summary.symbols,
            source_path=source_path,
            source_mtime_ns=stat.st_mtime_ns,
            source_size=stat.st_size,
            store_cells=summary.has_cell,
            progressive=True,
            temporary_cache=temporary_cache,
        )
        for frame_index in range(summary.frame_count):
            if cancel_event is not None and cancel_event.is_set():
                raise ConversionCancelled(f"Cancelled conversion for {source_path}")
            direct.read_frame_into(
                frame_index,
                store.positions[frame_index],
                None if store.cells is None else store.cells[frame_index],
            )
            store.mark_frame_available(frame_index + 1)
            if frame_index == 0 and preview_callback is not None:
                store.flush()
                preview_callback(store)
            if progress_callback is not None:
                progress_callback(frame_index + 1, summary.frame_count)
        store.mark_complete()
    except Exception:
        if store is not None:
            store.close()
        raise
    finally:
        direct.close()
    assert store is not None
    return store


def _coerce_source(source_value: Path | TrajectorySource) -> TrajectorySource:
    if isinstance(source_value, TrajectorySource):
        return source_value
    return TrajectorySource(Path(source_value).expanduser().resolve())
