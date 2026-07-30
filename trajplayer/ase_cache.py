from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

from .binary_store import BinaryTrajectoryStore, cache_dir_for_source
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
    if source_path.suffix.lower() == ".traj":
        from ase.io.trajectory import Trajectory

        traj = Trajectory(str(source_path))
        try:
            frame_count = len(traj)
            if frame_count <= 0:
                raise ValueError("No frames found in trajectory")
            first = traj[0]
            return _summary_from_first(first, frame_count)
        finally:
            traj.close()

    from ase.io import iread

    iterator = iread(str(source_path), index=":")
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("No frames found in trajectory") from exc

    frame_count = 1
    atom_count = len(first)
    atom_numbers = np.asarray(first.get_atomic_numbers(), dtype=np.uint16)
    for atoms in iterator:
        _validate_atoms(atoms, atom_count, atom_numbers, frame_count)
        frame_count += 1
    return _summary_from_first(first, frame_count)


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
    summary = inspect_ase_source(source_path)
    root = cache_root.resolve() if cache_root is not None else cache_dir_for_source(source_path)
    if root.exists():
        shutil.rmtree(root)

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
    )

    try:
        for frame_index, atoms in enumerate(_iter_frames(source_path)):
            if cancel_event is not None and cancel_event.is_set():
                raise ConversionCancelled(f"Cancelled conversion for {source_path}")
            _validate_atoms(atoms, summary.atom_count, summary.atom_numbers, frame_index)
            store.positions[frame_index, :, :] = np.asarray(atoms.get_positions(), dtype=np.float32)
            if store.cells is not None:
                store.cells[frame_index, :, :] = np.asarray(atoms.cell.array, dtype=np.float32)
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

    return store


def _summary_from_first(first: Atoms, frame_count: int) -> AseTrajectorySummary:
    return AseTrajectorySummary(
        frame_count=int(frame_count),
        atom_count=len(first),
        atom_numbers=np.asarray(first.get_atomic_numbers(), dtype=np.uint16),
        symbols=list(first.get_chemical_symbols()),
        has_cell=bool(np.any(np.asarray(first.cell.array, dtype=np.float32))),
    )


def _iter_frames(source_path: Path) -> Iterable[Atoms]:
    if source_path.suffix.lower() == ".traj":
        from ase.io.trajectory import Trajectory

        traj = Trajectory(str(source_path))
        try:
            for frame_index in range(len(traj)):
                yield traj[frame_index]
        finally:
            traj.close()
        return

    from ase.io import iread

    yield from iread(str(source_path), index=":")


def _validate_atoms(
    atoms: Atoms,
    expected_atom_count: int,
    expected_atom_numbers: np.ndarray,
    frame_index: int,
) -> None:
    if len(atoms) != expected_atom_count:
        raise ValueError(
            f"Frame {frame_index} has {len(atoms)} atoms; expected {expected_atom_count}"
        )
    atom_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.uint16)
    if not np.array_equal(atom_numbers, expected_atom_numbers):
        raise ValueError(f"Frame {frame_index} atom ordering differs from the first frame")


def _coerce_source(source_value: Path | TrajectorySource) -> TrajectorySource:
    if isinstance(source_value, TrajectorySource):
        return source_value
    return TrajectorySource(Path(source_value).expanduser().resolve())
