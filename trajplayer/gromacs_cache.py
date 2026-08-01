from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .binary_store import BinaryTrajectoryStore, cache_dir_for_source, prepare_cache_directory
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
    from ase.io import read

    topology_path, trajectory_path = _validated_paths(source)
    topology_atoms = read(str(topology_path), format="gromacs")
    universe = _open_universe(topology_path, trajectory_path)
    try:
        frame_count = len(universe.trajectory)
        if frame_count <= 0:
            raise ValueError("No frames found in Gromacs trajectory")
        if len(topology_atoms) != universe.atoms.n_atoms:
            raise ValueError(
                f"GRO topology has {len(topology_atoms)} atoms but {trajectory_path.name} has "
                f"{universe.atoms.n_atoms} atoms"
            )
        first_timestep = universe.trajectory[0]
        return GromacsTrajectorySummary(
            frame_count=frame_count,
            atom_count=len(topology_atoms),
            atom_numbers=np.asarray(topology_atoms.get_atomic_numbers(), dtype=np.uint16),
            symbols=list(topology_atoms.get_chemical_symbols()),
            has_cell=_cell_from_timestep(first_timestep) is not None,
        )
    finally:
        universe.trajectory.close()


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

    universe = _open_universe(topology_path, trajectory_path)
    try:
        for frame_index, timestep in enumerate(universe.trajectory):
            if cancel_event is not None and cancel_event.is_set():
                from .ase_cache import ConversionCancelled

                raise ConversionCancelled(f"Cancelled conversion for {trajectory_path}")
            positions = np.asarray(timestep.positions, dtype=np.float32)
            if positions.shape != (summary.atom_count, 3):
                raise ValueError(
                    f"Frame {frame_index} has position shape {positions.shape}; "
                    f"expected {(summary.atom_count, 3)}"
                )
            store.positions[frame_index, :, :] = positions
            if store.cells is not None:
                cell = _cell_from_timestep(timestep)
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
        universe.trajectory.close()
    return store


def _validated_paths(source: TrajectorySource) -> tuple[Path, Path]:
    if not source.is_gromacs_trajectory or source.topology_path is None:
        raise ValueError("A Gromacs trajectory requires one GRO topology and one XTC or TRR file")
    topology_path = source.topology_path.resolve()
    trajectory_path = source.trajectory_path.resolve()
    return topology_path, trajectory_path


def _open_universe(topology_path: Path, trajectory_path: Path):
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError("Gromacs XTC/TRR support requires a complete MDAnalysis installation") from exc
    return mda.Universe(str(topology_path), str(trajectory_path), in_memory=False)


def _cell_from_timestep(timestep) -> np.ndarray | None:
    cell = getattr(timestep, "triclinic_dimensions", None)
    if cell is None:
        return None
    matrix = np.ascontiguousarray(cell, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.any(matrix):
        return None
    return matrix
