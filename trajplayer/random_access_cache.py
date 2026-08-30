from __future__ import annotations

import hashlib
import mmap
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .binary_store import (
    BinaryTrajectoryStore,
    CacheValidationError,
    SourceIdentity,
    cache_dir_for_source,
    prepare_cache_directory,
)
from .ase_traj_reader import AseUlmTrajectoryReader
from .structure_reader import read_structure
from .trajcore import NATIVE_XYZ_READ_AVAILABLE, xyz_read_frame_into
from .trajectory_source import TrajectorySource
from .xyz_reader import inspect_xyz_frame_buffer, read_xyz_frame
from .xyz_index import FRAME_OFFSETS_FILE, IndexIoCoordinator, ProgressiveXyzIndex


RANDOM_ACCESS_SUFFIXES = frozenset(
    {".traj", ".xyz", ".extxyz", ".gro", ".pdb", ".cif", ".xtc", ".trr"}
)


@dataclass
class RandomAccessSummary:
    frame_count: int
    atom_count: int
    atom_numbers: np.ndarray
    symbols: list[str]
    has_cell: bool
    frame_count_is_final: bool = True


class RandomAccessFrameReader(Protocol):
    summary: RandomAccessSummary

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]: ...

    def wait_for_index_update(
        self,
        previous_count: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bool]: ...

    def close(self) -> None: ...


class RandomAccessTrajectoryStore:
    """Direct reader-backed frame source with no mandatory decoded sidecar."""

    def __init__(
        self,
        source: TrajectorySource,
        reader: RandomAccessFrameReader,
        *,
        root: Path,
    ) -> None:
        self.trajectory_source = source
        self.reader = reader
        self.root = root.resolve()
        self.atom_numbers = np.ascontiguousarray(reader.summary.atom_numbers, dtype=np.uint16)
        self.metadata: dict[str, object] = {
            "random_access": True,
            "direct_reader": True,
            "direct_read_into": callable(getattr(reader, "read_frame_into", None)),
            "native_xyz_read_into": bool(
                getattr(reader, "native_xyz_read_available", False)
            ),
            "persistent_decoded_cache": False,
            "source_files": [
                SourceIdentity.from_path(
                    path,
                    path.stat().st_mtime_ns,
                    path.stat().st_size,
                ).to_json()
                for path in source.paths
            ],
        }
        self._read_lock = threading.Lock()
        self._closed = False

    @property
    def frame_count(self) -> int:
        return int(self.reader.summary.frame_count)

    @property
    def frame_count_is_final(self) -> bool:
        return bool(self.reader.summary.frame_count_is_final)

    @property
    def atom_count(self) -> int:
        return int(self.reader.summary.atom_count)

    @property
    def has_cells(self) -> bool:
        return bool(self.reader.summary.has_cell)

    @property
    def supports_random_access(self) -> bool:
        return True

    @property
    def is_complete(self) -> bool:
        return self.frame_count_is_final

    @property
    def available_frame_count(self) -> int:
        return self.frame_count

    @property
    def navigable_frame_count(self) -> int:
        return self.frame_count

    def is_frame_available(self, frame_index: int) -> bool:
        return 0 <= int(frame_index) < self.frame_count

    def read_frame_arrays(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        with self._read_lock:
            positions, cell = self.reader.read_frame(int(frame_index))
        return (
            np.ascontiguousarray(positions, dtype=np.float32),
            None if cell is None else np.ascontiguousarray(cell, dtype=np.float32),
        )

    def read_frame_into(
        self,
        frame_index: int,
        positions: np.ndarray,
        cell: np.ndarray | None,
    ) -> None:
        direct_read = getattr(self.reader, "read_frame_into", None)
        if callable(direct_read) and _is_direct_frame_destination(
            positions,
            cell,
            atom_count=self.atom_count,
        ):
            with self._read_lock:
                direct_read(int(frame_index), positions, cell)
            return
        source_positions, source_cell = self.read_frame_arrays(frame_index)
        np.copyto(positions, source_positions)
        if cell is not None:
            if source_cell is None:
                cell.fill(0.0)
            else:
                np.copyto(cell, source_cell)

    def frame(self, frame_index: int) -> np.ndarray:
        positions, _cell = self.read_frame_arrays(frame_index)
        return positions

    def cell(self, frame_index: int) -> np.ndarray | None:
        _positions, cell = self.read_frame_arrays(frame_index)
        return cell

    def wait_for_index_update(
        self,
        previous_count: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bool]:
        return self.reader.wait_for_index_update(
            int(previous_count),
            timeout_s=float(timeout_s),
        )

    def close(self) -> None:
        with self._read_lock:
            if self._closed:
                return
            self._closed = True
            self.reader.close()


def index_cache_dir_for_source(source_path: Path) -> Path:
    path = source_path.resolve()
    adjacent = path.with_name(f"{path.name}.tpindex")
    try:
        adjacent.mkdir(parents=True, exist_ok=True)
        return adjacent
    except OSError:
        cache_base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("XDG_CACHE_HOME")
            or (Path.home() / ".cache")
        )
        digest = hashlib.sha256(os.fsencode(str(path))).hexdigest()[:20]
        fallback = cache_base / "TrajPlayer" / "indexes" / digest
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def supports_random_access_source(source: TrajectorySource) -> bool:
    return source.trajectory_path.suffix.lower() in RANDOM_ACCESS_SUFFIXES


def open_direct_random_access_store(
    source: TrajectorySource,
    *,
    status_callback: Callable[[str], None] | None = None,
    index_progress_callback: Callable[[int, bool], None] | None = None,
) -> RandomAccessTrajectoryStore:
    if source.trajectory_path.suffix.lower() in {".xyz", ".extxyz"}:
        root = index_cache_dir_for_source(source.trajectory_path)
    else:
        root = _reader_state_dir_for_source(source.trajectory_path)
    reader = _open_reader(
        source,
        root,
        status_callback=status_callback,
        index_progress_callback=index_progress_callback,
    )
    return RandomAccessTrajectoryStore(source, reader, root=root)


def open_random_access_session(
    source: TrajectorySource,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[RandomAccessFrameReader, BinaryTrajectoryStore]:
    canonical_root = cache_dir_for_source(source.trajectory_path)
    root = canonical_root
    temporary_cache = False
    store = _open_compatible_partial_store(source, canonical_root)
    if store is None:
        root, temporary_cache = prepare_cache_directory(canonical_root)

    try:
        reader = _open_reader(
            source,
            root,
            status_callback=status_callback,
            index_progress_callback=None,
        )
        while not reader.summary.frame_count_is_final:
            reader.wait_for_index_update(reader.summary.frame_count, timeout_s=0.05)
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
            reader = _open_reader(
                source,
                root,
                status_callback=status_callback,
                index_progress_callback=None,
            )
            while not reader.summary.frame_count_is_final:
                reader.wait_for_index_update(reader.summary.frame_count, timeout_s=0.05)
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
    direct_read = getattr(reader, "read_frame_into", None)
    if callable(direct_read):
        direct_read(
            index,
            store.positions[index],
            None if store.cells is None else store.cells[index],
        )
        store.publish_frame(index)
        return
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
        self._trajectory = AseUlmTrajectoryReader(path)
        self.summary = RandomAccessSummary(
            frame_count=self._trajectory.frame_count,
            atom_count=self._trajectory.atom_count,
            atom_numbers=self._trajectory.atom_numbers,
            symbols=list(self._trajectory.symbols),
            has_cell=self._trajectory.has_cell,
        )

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        return self._trajectory.read_frame(int(frame_index))

    def wait_for_index_update(
        self,
        previous_count: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bool]:
        del previous_count, timeout_s
        return self.summary.frame_count, True

    def close(self) -> None:
        self._trajectory.close()


class _GromacsReader:
    def __init__(self, source: TrajectorySource) -> None:
        if source.topology_path is None:
            raise ValueError("A Gromacs trajectory requires a GRO topology")
        from .gromacs_reader import ChemfilesGromacsReader

        topology = read_structure(source.topology_path)
        self._reader = ChemfilesGromacsReader(
            source.trajectory_path,
            expected_atom_count=topology.positions.shape[0],
        )
        self.summary = RandomAccessSummary(
            frame_count=self._reader.frame_count,
            atom_count=topology.positions.shape[0],
            atom_numbers=topology.atom_numbers,
            symbols=list(topology.symbols),
            has_cell=self._reader.has_cell,
        )

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        return self._reader.read_frame(int(frame_index))

    def wait_for_index_update(
        self,
        previous_count: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bool]:
        del previous_count, timeout_s
        return self.summary.frame_count, True

    def close(self) -> None:
        self._reader.close()


class _IndexedXyzReader:
    def __init__(
        self,
        path: Path,
        cache_root: Path,
        *,
        status_callback: Callable[[str], None] | None,
        index_progress_callback: Callable[[int, bool], None] | None,
    ) -> None:
        self._path = path.resolve()
        self._file = self._path.open("rb")
        self._mapped_file: mmap.mmap | None = None
        self._prefetched_first_frame = None
        self._native_direct_enabled = bool(NATIVE_XYZ_READ_AVAILABLE)
        self.native_read_count = 0
        self.python_read_count = 0
        self.prefetched_read_count = 0
        with self._path.open("rb") as source_handle:
            header = source_handle.readline()
        try:
            atom_count = int(header.strip())
        except ValueError as exc:
            self._file.close()
            raise ValueError("XYZ trajectory does not start with an atom-count line") from exc
        if atom_count <= 0:
            self._file.close()
            raise ValueError("XYZ atom count must be positive")
        self._mapped_file = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)

        def publish_progress(frame_count: int, complete: bool) -> None:
            self.summary.frame_count = int(frame_count)
            self.summary.frame_count_is_final = bool(complete)
            if index_progress_callback is not None:
                index_progress_callback(int(frame_count), bool(complete))

        self._io_coordinator = IndexIoCoordinator()
        try:
            self._index = ProgressiveXyzIndex(
                self._path,
                cache_root,
                atom_count=atom_count,
                progress_callback=publish_progress,
                io_coordinator=self._io_coordinator,
            )
            with self._io_coordinator.foreground():
                first = self._read_frame(0, atom_count=atom_count)
            self.summary = RandomAccessSummary(
                frame_count=self._index.known_frame_count,
                atom_count=atom_count,
                atom_numbers=first.atom_numbers,
                symbols=list(first.symbols),
                has_cell=first.cell is not None,
            )
            self._prefetched_first_frame = first
            self.summary.frame_count_is_final = self._index.complete
            if not self._index.complete and status_callback is not None:
                status_callback(
                    f"Opened first frame; indexing {self._path.name} in the background"
                )
            self._index.start()
        except Exception:
            index = getattr(self, "_index", None)
            if index is not None:
                index.close()
            if self._mapped_file is not None:
                self._mapped_file.close()
                self._mapped_file = None
            self._file.close()
            raise

    @property
    def native_xyz_read_available(self) -> bool:
        return bool(NATIVE_XYZ_READ_AVAILABLE)

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        index = int(frame_index)
        with self._io_coordinator.foreground():
            frame = self._take_prefetched_frame(index)
            if frame is None:
                frame = self._read_frame(index)
                self.python_read_count += 1
        if not np.array_equal(frame.atom_numbers, self.summary.atom_numbers):
            raise ValueError(f"Frame {index} atom ordering differs from the first frame")
        return frame.positions, frame.cell

    def read_frame_into(
        self,
        frame_index: int,
        positions: np.ndarray,
        cell: np.ndarray | None,
    ) -> None:
        index = int(frame_index)
        expected_shape = (self.summary.atom_count, 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"Frame destination has position shape {positions.shape}; expected {expected_shape}"
            )
        with self._io_coordinator.foreground():
            prefetched = self._take_prefetched_frame(index)
            if prefetched is not None:
                self._copy_frame_into(prefetched, positions, cell, frame_index=index)
                return
            if index < 0 or index >= self._index.known_frame_count:
                raise IndexError(index)
            if self._native_direct_enabled and self._mapped_file is not None:
                start = self._index.offset(index)
                frame_end = (
                    self._index.offset(index + 1)
                    if index + 1 < self._index.known_frame_count
                    else len(self._mapped_file)
                )
                try:
                    layout = inspect_xyz_frame_buffer(
                        self._mapped_file,
                        start,
                        self.summary.atom_count,
                        index,
                    )
                    used_native = xyz_read_frame_into(
                        self._mapped_file,
                        data_offset=layout.data_offset,
                        data_end=frame_end,
                        positions=positions,
                        identity_column=layout.identity_column,
                        identity_is_atomic_number=layout.identity_is_atomic_number,
                        position_columns=layout.position_columns,
                        expected_columns=layout.expected_columns,
                        expected_atom_numbers=self.summary.atom_numbers,
                    )
                    if used_native:
                        self._copy_cell_into(layout.cell, cell)
                        self.native_read_count += 1
                        return
                    self._native_direct_enabled = False
                except ValueError as native_error:
                    try:
                        fallback = self._read_frame(index)
                        self._validate_atom_order(fallback, index)
                    except Exception as fallback_error:
                        raise fallback_error from native_error
                    self._native_direct_enabled = False
                    self.python_read_count += 1
                    self._copy_frame_into(fallback, positions, cell, frame_index=index)
                    return

            frame = self._read_frame(index)
            self._validate_atom_order(frame, index)
            self.python_read_count += 1
            self._copy_frame_into(frame, positions, cell, frame_index=index)

    def wait_for_index_update(
        self,
        previous_count: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bool]:
        frame_count, complete = self._index.wait_for_update(
            int(previous_count),
            timeout_s=float(timeout_s),
        )
        self.summary.frame_count = int(frame_count)
        self.summary.frame_count_is_final = bool(complete)
        return frame_count, complete

    def close(self) -> None:
        self._index.close()
        self._prefetched_first_frame = None
        if self._mapped_file is not None:
            self._mapped_file.close()
            self._mapped_file = None
        self._file.close()

    def _read_frame(self, frame_index: int, *, atom_count: int | None = None):
        if frame_index < 0 or frame_index >= self._index.known_frame_count:
            raise IndexError(frame_index)
        start = self._index.offset(frame_index)
        self._file.seek(start)
        expected_atoms = self._index.atom_count if atom_count is None else int(atom_count)
        return read_xyz_frame(self._file, expected_atoms, frame_index)

    def _take_prefetched_frame(self, frame_index: int):
        if int(frame_index) != 0 or self._prefetched_first_frame is None:
            return None
        frame = self._prefetched_first_frame
        self._prefetched_first_frame = None
        self.prefetched_read_count += 1
        return frame

    def _validate_atom_order(self, frame, frame_index: int) -> None:
        if not np.array_equal(frame.atom_numbers, self.summary.atom_numbers):
            raise ValueError(f"Frame {frame_index} atom ordering differs from the first frame")

    def _copy_frame_into(
        self,
        frame,
        positions: np.ndarray,
        cell: np.ndarray | None,
        *,
        frame_index: int,
    ) -> None:
        self._validate_atom_order(frame, frame_index)
        np.copyto(positions, frame.positions)
        self._copy_cell_into(frame.cell, cell)

    @staticmethod
    def _copy_cell_into(source_cell: np.ndarray | None, cell: np.ndarray | None) -> None:
        if cell is None:
            return
        if source_cell is None:
            cell.fill(0.0)
        else:
            np.copyto(cell, source_cell)


class _StaticStructureReader:
    def __init__(self, path: Path) -> None:
        self._frame = read_structure(path)
        self.summary = RandomAccessSummary(
            frame_count=1,
            atom_count=self._frame.positions.shape[0],
            atom_numbers=self._frame.atom_numbers,
            symbols=list(self._frame.symbols),
            has_cell=self._frame.cell is not None,
        )

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        if int(frame_index) != 0:
            raise IndexError(frame_index)
        return self._frame.positions, self._frame.cell

    def wait_for_index_update(
        self,
        previous_count: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bool]:
        del previous_count, timeout_s
        return 1, True

    def close(self) -> None:
        self._frame = None


def _open_reader(
    source: TrajectorySource,
    cache_root: Path,
    *,
    status_callback: Callable[[str], None] | None,
    index_progress_callback: Callable[[int, bool], None] | None,
) -> RandomAccessFrameReader:
    suffix = source.trajectory_path.suffix.lower()
    if suffix == ".traj":
        return _AseTrajectoryReader(source.trajectory_path)
    if suffix in {".xyz", ".extxyz"}:
        return _IndexedXyzReader(
            source.trajectory_path,
            cache_root,
            status_callback=status_callback,
            index_progress_callback=index_progress_callback,
        )
    if suffix in {".xtc", ".trr"}:
        return _GromacsReader(source)
    if suffix in {".gro", ".pdb", ".cif"}:
        return _StaticStructureReader(source.trajectory_path)
    raise ValueError(f"Random frame access is not supported for {source.trajectory_path.name}")


def _open_compatible_partial_store(
    source: TrajectorySource,
    root: Path,
) -> BinaryTrajectoryStore | None:
    try:
        store = BinaryTrajectoryStore.open(root, mode="r+")
    except (CacheValidationError, FileNotFoundError):
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


def _reader_state_dir_for_source(source_path: Path) -> Path:
    path = source_path.resolve()
    cache_base = Path(
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("XDG_CACHE_HOME")
        or (Path.home() / ".cache")
    )
    digest = hashlib.sha256(os.fsencode(str(path))).hexdigest()[:20]
    return cache_base / "TrajPlayer" / "readers" / digest


def _is_direct_frame_destination(
    positions: np.ndarray,
    cell: np.ndarray | None,
    *,
    atom_count: int,
) -> bool:
    if not isinstance(positions, np.ndarray):
        return False
    if (
        positions.dtype != np.float32
        or positions.shape != (int(atom_count), 3)
        or not positions.flags.c_contiguous
        or not positions.flags.writeable
    ):
        return False
    if cell is None:
        return True
    return bool(
        isinstance(cell, np.ndarray)
        and cell.dtype == np.float32
        and cell.shape == (3, 3)
        and cell.flags.c_contiguous
        and cell.flags.writeable
    )
