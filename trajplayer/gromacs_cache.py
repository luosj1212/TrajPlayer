from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .binary_store import BinaryTrajectoryStore, cache_dir_for_source, prepare_cache_directory
from .gromacs_reader import ChemfilesGromacsReader
from .structure_reader import read_structure
from .trajectory_source import TrajectorySource


ProgressCallback = Callable[[int, int], None]
PreviewCallback = Callable[[BinaryTrajectoryStore], None]


@dataclass(frozen=True)
class GromacsTrajectorySummary:
    frame_count: int
    atom_count: int
    atom_numbers: np.ndarray
    symbols: list[str]
    has_cell: bool


def inspect_gromacs_source(source: TrajectorySource) -> GromacsTrajectorySummary:
    topology_path, trajectory_path = _validated_paths(source)
    topology = read_structure(topology_path)
    reader = ChemfilesGromacsReader(
        trajectory_path,
        expected_atom_count=topology.positions.shape[0],
    )
    try:
        return GromacsTrajectorySummary(
            frame_count=reader.frame_count,
            atom_count=topology.positions.shape[0],
            atom_numbers=topology.atom_numbers,
            symbols=list(topology.symbols),
            has_cell=reader.has_cell,
        )
    finally:
        reader.close()


def build_cache_from_gromacs(
    source: TrajectorySource,
    *,
    cache_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> BinaryTrajectoryStore:
    topology_path, trajectory_path = _validated_paths(source)
    summary = inspect_gromacs_source(source)
    root = cache_root.resolve() if cache_root is not None else cache_dir_for_source(trajectory_path)
    root, temporary_cache = prepare_cache_directory(root)

    trajectory_stat = trajectory_path.stat()
    store = BinaryTrajectoryStore.create(
        root,
        frame_count=summary.frame_count,
        atom_numbers=summary.atom_numbers,
        symbols=summary.symbols,
        source_path=trajectory_path,
        source_mtime_ns=trajectory_stat.st_mtime_ns,
        source_size=trajectory_stat.st_size,
        source_paths=source.paths,
        store_cells=summary.has_cell,
        progressive=True,
        temporary_cache=temporary_cache,
    )

    reader = ChemfilesGromacsReader(
        trajectory_path,
        expected_atom_count=summary.atom_count,
    )
    try:
        for frame_index in range(summary.frame_count):
            if cancel_event is not None and cancel_event.is_set():
                from .ase_cache import ConversionCancelled

                raise ConversionCancelled(f"Cancelled conversion for {trajectory_path}")
            positions, cell = reader.read_frame(frame_index)
            if positions.shape != (summary.atom_count, 3):
                raise ValueError(
                    f"Frame {frame_index} has position shape {positions.shape}; "
                    f"expected {(summary.atom_count, 3)}"
                )
            store.positions[frame_index, :, :] = positions
            if store.cells is not None:
                store.cells[frame_index, :, :] = 0.0 if cell is None else cell
            store.mark_frame_available(frame_index + 1)
            if frame_index == 0 and preview_callback is not None:
                store.flush()
                preview_callback(store)
            if progress_callback is not None:
                progress_callback(frame_index + 1, summary.frame_count)
        store.mark_complete()
    except Exception:
        store.close()
        raise
    finally:
        reader.close()
    return store


def _validated_paths(source: TrajectorySource) -> tuple[Path, Path]:
    if not source.is_gromacs_trajectory or source.topology_path is None:
        raise ValueError("A Gromacs trajectory requires one GRO topology and one XTC or TRR file")
    topology_path = source.topology_path.resolve()
    trajectory_path = source.trajectory_path.resolve()
    return topology_path, trajectory_path
