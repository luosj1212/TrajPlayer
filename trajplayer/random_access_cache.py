from __future__ import annotations

import io
import json
import mmap
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .binary_store import (
    BinaryTrajectoryStore,
    SourceIdentity,
    cache_dir_for_source,
    prepare_cache_directory,
)
from .trajectory_source import TrajectorySource


RANDOM_ACCESS_SUFFIXES = frozenset({".traj", ".xyz", ".extxyz", ".xtc", ".trr"})
FRAME_OFFSETS_FILE = "frame_offsets.u64"
FRAME_INDEX_METADATA_FILE = "frame_index.json"
INDEX_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RandomAccessSummary:
    frame_count: int
    atom_count: int
    atom_numbers: np.ndarray
    symbols: list[str]
    has_cell: bool


class RandomAccessFrameReader(Protocol):
    summary: RandomAccessSummary

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]: ...

    def close(self) -> None: ...


def supports_random_access_source(source: TrajectorySource) -> bool:
    return source.trajectory_path.suffix.lower() in RANDOM_ACCESS_SUFFIXES


def open_random_access_session(
    source: TrajectorySource,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[RandomAccessFrameReader, BinaryTrajectoryStore]:
    canonical_root = cache_dir_for_source(source.trajectory_path).resolve()
    root = canonical_root
    temporary_cache = False
    store = _open_compatible_partial_store(source, canonical_root)
    if store is None:
        root, temporary_cache = prepare_cache_directory(canonical_root)

    try:
        reader = _open_reader(source, root, status_callback=status_callback)
    except Exception:
        if store is not None:
            store.close()
        if temporary_cache:
            shutil.rmtree(root, ignore_errors=True)
        raise

    summary = reader.summary
    if store is not None and not _store_matches_summary(store, summary):
        store.close()
        reader.close()
        root, temporary_cache = prepare_cache_directory(canonical_root)
        try:
            reader = _open_reader(source, root, status_callback=status_callback)
        except Exception:
            if temporary_cache:
                shutil.rmtree(root, ignore_errors=True)
            raise
        summary = reader.summary
        store = None

    if store is None:
        stat = source.trajectory_path.stat()
        store = BinaryTrajectoryStore.create(
            root,
            frame_count=summary.frame_count,
            atom_numbers=summary.atom_numbers,
            symbols=summary.symbols,
            source_path=source.trajectory_path,
            source_mtime_ns=stat.st_mtime_ns,
            source_size=stat.st_size,
            source_paths=source.paths,
            store_cells=summary.has_cell,
            progressive=True,
            random_access=True,
            temporary_cache=temporary_cache,
        )
    return reader, store


def write_reader_frame(
    reader: RandomAccessFrameReader,
    store: BinaryTrajectoryStore,
    frame_index: int,
) -> None:
    index = int(frame_index)
    positions, cell = reader.read_frame(index)
    frame = np.asarray(positions, dtype=np.float32)
    expected_shape = (store.atom_count, 3)
    if frame.shape != expected_shape:
        raise ValueError(f"Frame {index} has position shape {frame.shape}; expected {expected_shape}")
    np.copyto(store.positions[index], frame)
    if store.cells is not None:
        if cell is None:
            store.cells[index, :, :] = 0.0
        else:
            matrix = np.asarray(cell, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError(f"Frame {index} has cell shape {matrix.shape}; expected (3, 3)")
            np.copyto(store.cells[index], matrix)
    store.publish_frame(index)


class _AseTrajectoryReader:
    def __init__(self, path: Path) -> None:
        from ase.io.trajectory import Trajectory

        self._trajectory = Trajectory(str(path))
        frame_count = len(self._trajectory)
        if frame_count <= 0:
            self._trajectory.close()
            raise ValueError("No frames found in trajectory")
        first = self._trajectory[0]
        self.summary = _summary_from_atoms(first, frame_count)

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        atoms = self._trajectory[int(frame_index)]
        _validate_atom_layout(atoms, self.summary, int(frame_index))
        return _arrays_from_atoms(atoms)

    def close(self) -> None:
        self._trajectory.close()


class _GromacsReader:
    def __init__(self, source: TrajectorySource) -> None:
        if source.topology_path is None:
            raise ValueError("A Gromacs trajectory requires a GRO topology")
        from ase.io import read

        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise RuntimeError(
                "Gromacs XTC/TRR support requires a complete MDAnalysis installation"
            ) from exc

        topology_atoms = read(str(source.topology_path), format="gromacs")
        self._universe = mda.Universe(
            str(source.topology_path),
            str(source.trajectory_path),
            in_memory=False,
        )
        frame_count = len(self._universe.trajectory)
        if frame_count <= 0:
            self.close()
            raise ValueError("No frames found in Gromacs trajectory")
        if len(topology_atoms) != self._universe.atoms.n_atoms:
            self.close()
            raise ValueError(
                f"GRO topology has {len(topology_atoms)} atoms but {source.trajectory_path.name} "
                f"has {self._universe.atoms.n_atoms} atoms"
            )
        first = self._universe.trajectory[0]
        self.summary = RandomAccessSummary(
            frame_count=frame_count,
            atom_count=len(topology_atoms),
            atom_numbers=np.asarray(topology_atoms.get_atomic_numbers(), dtype=np.uint16),
            symbols=list(topology_atoms.get_chemical_symbols()),
            has_cell=_cell_from_timestep(first) is not None,
        )

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        timestep = self._universe.trajectory[int(frame_index)]
        return (
            np.ascontiguousarray(timestep.positions, dtype=np.float32),
            _cell_from_timestep(timestep),
        )

    def close(self) -> None:
        self._universe.trajectory.close()


class _IndexedXyzReader:
    def __init__(
        self,
        path: Path,
        cache_root: Path,
        *,
        status_callback: Callable[[str], None] | None,
    ) -> None:
        self._path = path.resolve()
        self._file = self._path.open("rb")
        self._offsets_path = cache_root / FRAME_OFFSETS_FILE
        metadata_path = cache_root / FRAME_INDEX_METADATA_FILE
        if not _index_is_valid(metadata_path, self._path, self._offsets_path):
            if status_callback is not None:
                status_callback(f"Indexing {self._path.name} without parsing atom records")
            frame_count, atom_count = _build_xyz_index(
                self._path,
                self._offsets_path,
            )
            identity = SourceIdentity.from_path(
                self._path,
                self._path.stat().st_mtime_ns,
                self._path.stat().st_size,
            )
            _write_json_atomic(
                metadata_path,
                {
                    "source": identity.to_json(),
                    "frame_count": frame_count,
                    "atom_count": atom_count,
                },
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_count = int(metadata["frame_count"])
        atom_count = int(metadata["atom_count"])
        self._offsets = np.memmap(
            self._offsets_path,
            dtype=np.uint64,
            mode="r",
            shape=(frame_count,),
        )
        first = self._read_atoms(0)
        if len(first) != atom_count:
            raise ValueError("XYZ frame index atom count does not match the first frame")
        self.summary = _summary_from_atoms(first, frame_count)

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        index = int(frame_index)
        atoms = self._read_atoms(index)
        _validate_atom_layout(atoms, self.summary, index)
        return _arrays_from_atoms(atoms)

    def close(self) -> None:
        mmap_obj = getattr(self._offsets, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
        self._file.close()

    def _read_atoms(self, frame_index: int):
        from ase.io.extxyz import _read_xyz_frame, key_val_str_to_dict

        if frame_index < 0 or frame_index >= len(self._offsets):
            raise IndexError(frame_index)
        start = int(self._offsets[frame_index])
        end = (
            int(self._offsets[frame_index + 1])
            if frame_index + 1 < len(self._offsets)
            else self._path.stat().st_size
        )
        self._file.seek(start)
        text = io.StringIO(self._file.read(end - start).decode("utf-8", errors="replace"))
        try:
            atom_count = int(text.readline().strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ frame header at frame {frame_index}") from exc
        return _read_xyz_frame(text, atom_count, key_val_str_to_dict, nvec=0)


def _open_reader(
    source: TrajectorySource,
    cache_root: Path,
    *,
    status_callback: Callable[[str], None] | None,
) -> RandomAccessFrameReader:
    suffix = source.trajectory_path.suffix.lower()
    if suffix == ".traj":
        return _AseTrajectoryReader(source.trajectory_path)
    if suffix in {".xyz", ".extxyz"}:
        return _IndexedXyzReader(
            source.trajectory_path,
            cache_root,
            status_callback=status_callback,
        )
    if suffix in {".xtc", ".trr"}:
        return _GromacsReader(source)
    raise ValueError(f"Random frame access is not supported for {source.trajectory_path.name}")


def _open_compatible_partial_store(
    source: TrajectorySource,
    root: Path,
) -> BinaryTrajectoryStore | None:
    try:
        store = BinaryTrajectoryStore.open(root)
    except Exception:
        return None
    if not store.supports_random_access or not store.is_valid_for_sources(source.paths):
        store.close()
        return None
    return store


def _store_matches_summary(
    store: BinaryTrajectoryStore,
    summary: RandomAccessSummary,
) -> bool:
    return (
        store.frame_count == summary.frame_count
        and store.atom_count == summary.atom_count
        and store.has_cells == summary.has_cell
        and np.array_equal(store.atom_numbers, summary.atom_numbers)
    )


def _summary_from_atoms(atoms, frame_count: int) -> RandomAccessSummary:
    return RandomAccessSummary(
        frame_count=int(frame_count),
        atom_count=len(atoms),
        atom_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.uint16),
        symbols=list(atoms.get_chemical_symbols()),
        has_cell=bool(np.any(np.asarray(atoms.cell.array, dtype=np.float32))),
    )


def _validate_atom_layout(atoms, summary: RandomAccessSummary, frame_index: int) -> None:
    if len(atoms) != summary.atom_count:
        raise ValueError(
            f"Frame {frame_index} has {len(atoms)} atoms; expected {summary.atom_count}"
        )
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.uint16)
    if not np.array_equal(numbers, summary.atom_numbers):
        raise ValueError(f"Frame {frame_index} atom ordering differs from the first frame")


def _arrays_from_atoms(atoms) -> tuple[np.ndarray, np.ndarray | None]:
    positions = np.ascontiguousarray(atoms.get_positions(), dtype=np.float32)
    cell = np.asarray(atoms.cell.array, dtype=np.float32)
    return positions, np.ascontiguousarray(cell) if np.any(cell) else None


def _cell_from_timestep(timestep) -> np.ndarray | None:
    cell = getattr(timestep, "triclinic_dimensions", None)
    if cell is None:
        return None
    matrix = np.ascontiguousarray(cell, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.any(matrix):
        return None
    return matrix


def _build_xyz_index(source: Path, offsets_path: Path) -> tuple[int, int]:
    file_size = source.stat().st_size
    if file_size <= 0:
        raise ValueError("No frames found in trajectory")
    with source.open("rb") as handle:
        header = handle.readline()
    try:
        atom_count = int(header.strip())
    except ValueError as exc:
        raise ValueError("XYZ trajectory does not start with an atom-count line") from exc
    if atom_count <= 0:
        raise ValueError("XYZ atom count must be positive")

    lines_per_frame = atom_count + 2
    frame_count = 1
    with source.open("rb") as source_handle, offsets_path.open("wb") as output:
        output.write(np.uint64(0).tobytes())
        with mmap.mmap(source_handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            line_count = 0
            for chunk_start in range(0, file_size, INDEX_CHUNK_BYTES):
                chunk_size = min(INDEX_CHUNK_BYTES, file_size - chunk_start)
                chunk = np.frombuffer(mapped, dtype=np.uint8, count=chunk_size, offset=chunk_start)
                newlines = np.flatnonzero(chunk == 10)
                first_boundary = lines_per_frame - 1 - (line_count % lines_per_frame)
                if first_boundary < newlines.size:
                    boundaries = newlines[first_boundary::lines_per_frame]
                    offsets = boundaries.astype(np.uint64, copy=True)
                    offsets += np.uint64(chunk_start + 1)
                    offsets = offsets[offsets < file_size]
                    if offsets.size:
                        offsets.tofile(output)
                        frame_count += int(offsets.size)
                line_count += int(newlines.size)
                del newlines
                del chunk

    offsets_size = offsets_path.stat().st_size
    if offsets_size != frame_count * np.dtype(np.uint64).itemsize:
        raise RuntimeError("XYZ frame index size is inconsistent")
    return frame_count, atom_count


def _index_is_valid(metadata_path: Path, source: Path, offsets_path: Path) -> bool:
    if not metadata_path.exists() or not offsets_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stat = source.stat()
        expected = SourceIdentity.from_path(source, stat.st_mtime_ns, stat.st_size)
        stored = SourceIdentity.from_json(metadata["source"])
        frame_count = int(metadata["frame_count"])
        return (
            stored == expected
            and frame_count > 0
            and offsets_path.stat().st_size == frame_count * np.dtype(np.uint64).itemsize
        )
    except Exception:
        return False


def _write_json_atomic(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
