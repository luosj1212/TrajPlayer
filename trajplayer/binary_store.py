from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


STORE_VERSION = 3
SUPPORTED_STORE_VERSIONS = frozenset({2, STORE_VERSION})
POSITIONS_FILE = "positions.f32"
CELLS_FILE = "cells.f32"
ATOM_NUMBERS_FILE = "atom_numbers.u16"
FRAME_AVAILABILITY_FILE = "frame_availability.u8"
METADATA_FILE = "metadata.json"
MAX_METADATA_BYTES = 1024 * 1024


class CacheValidationError(ValueError):
    """Raised when a trajectory sidecar is unsafe or internally inconsistent."""


def cache_dir_for_source(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.name}.tpdata")


def prepare_cache_directory(root: Path) -> tuple[Path, bool]:
    """Create an empty cache directory without deleting a cache in active use."""
    root = root.resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return root, False

    if os.name != "nt":
        shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        return root, False

    retired = _unique_sibling(root, "retired")
    try:
        root.rename(retired)
    except OSError as exc:
        if getattr(exc, "winerror", None) not in {5, 32}:
            raise
        temporary = _unique_sibling(root, "session")
        temporary.mkdir(parents=True, exist_ok=False)
        return temporary, True

    try:
        root.mkdir(parents=True, exist_ok=False)
    except Exception:
        try:
            retired.rename(root)
        except OSError:
            pass
        raise
    try:
        shutil.rmtree(retired)
    except OSError:
        pass
    return root, False


def _unique_sibling(root: Path, purpose: str) -> Path:
    return root.with_name(
        f".{root.name}.{purpose}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )


@dataclass(frozen=True)
class SourceIdentity:
    path: str | None
    mtime_ns: int
    size: int

    @classmethod
    def from_path(cls, path: Path | None, mtime_ns: int = 0, size: int = 0) -> "SourceIdentity":
        if path is None:
            return cls(path=None, mtime_ns=mtime_ns, size=size)
        return cls(path=str(path.resolve()), mtime_ns=int(mtime_ns), size=int(size))

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "mtime_ns": self.mtime_ns, "size": self.size}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SourceIdentity":
        return cls(
            path=data.get("path"),
            mtime_ns=int(data.get("mtime_ns", 0)),
            size=int(data.get("size", 0)),
        )


class BinaryTrajectoryStore:
    """A trajectory stored as a single contiguous (T, N, 3) float32 memmap."""

    def __init__(
        self,
        root: Path,
        metadata: dict[str, Any],
        positions: np.memmap,
        atom_numbers: np.memmap,
        cells: np.memmap | None = None,
        frame_availability: np.memmap | None = None,
    ) -> None:
        self.root = root
        self.metadata = metadata
        self.positions = positions
        self.atom_numbers = atom_numbers
        self.cells = cells
        self.frame_availability = frame_availability
        self.source = SourceIdentity.from_json(metadata.get("source", {}))
        self._availability_lock = threading.Lock()
        if frame_availability is not None:
            self._available_frame_count = int(np.count_nonzero(frame_availability))
            unavailable = np.flatnonzero(frame_availability == 0)
            self._available_prefix_count = (
                self.frame_count if unavailable.size == 0 else int(unavailable[0])
            )
        else:
            self._available_frame_count = int(
                metadata.get(
                    "available_frame_count",
                    self.frame_count if metadata.get("complete", True) else 0,
                )
            )
            self._available_prefix_count = self._available_frame_count
        self._closed = False

    @property
    def frame_count(self) -> int:
        return int(self.metadata["frame_count"])

    @property
    def atom_count(self) -> int:
        return int(self.metadata["atom_count"])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.frame_count, self.atom_count, 3)

    @property
    def has_cells(self) -> bool:
        return self.cells is not None

    @property
    def is_complete(self) -> bool:
        return bool(self.metadata.get("complete", True))

    @property
    def supports_random_access(self) -> bool:
        return bool(self.metadata.get("random_access", False))

    @property
    def available_frame_count(self) -> int:
        with self._availability_lock:
            return self._available_frame_count

    @property
    def available_prefix_count(self) -> int:
        with self._availability_lock:
            return self._available_prefix_count

    @property
    def navigable_frame_count(self) -> int:
        if self.supports_random_access:
            return self.frame_count
        return self.available_prefix_count

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        frame_count: int,
        atom_numbers: np.ndarray,
        symbols: list[str] | None,
        source_path: Path | None,
        source_mtime_ns: int,
        source_size: int,
        source_paths: Sequence[Path] | None = None,
        store_cells: bool = False,
        progressive: bool = False,
        random_access: bool = False,
        temporary_cache: bool = False,
    ) -> "BinaryTrajectoryStore":
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

        atom_numbers_array = np.asarray(atom_numbers, dtype=np.uint16)
        if atom_numbers_array.ndim != 1:
            raise ValueError("atom_numbers must be a 1D array")
        atom_count = int(atom_numbers_array.shape[0])
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if atom_count <= 0:
            raise ValueError("atom_count must be positive")

        atom_numbers_array.tofile(root / ATOM_NUMBERS_FILE)
        shape = (int(frame_count), atom_count, 3)
        positions = np.memmap(root / POSITIONS_FILE, dtype=np.float32, mode="w+", shape=shape)
        cells = (
            np.memmap(root / CELLS_FILE, dtype=np.float32, mode="w+", shape=(int(frame_count), 3, 3))
            if store_cells
            else None
        )
        frame_availability = None
        if progressive:
            frame_availability = np.memmap(
                root / FRAME_AVAILABILITY_FILE,
                dtype=np.uint8,
                mode="w+",
                shape=(int(frame_count),),
            )
            frame_availability[:] = 0
            frame_availability.flush()

        metadata: dict[str, Any] = {
            "version": STORE_VERSION,
            "dtype": "float32",
            "frame_count": int(frame_count),
            "atom_count": atom_count,
            "shape": list(shape),
            "complete": not progressive,
            "available_frame_count": 0 if progressive else int(frame_count),
            "random_access": bool(random_access),
            "temporary_cache": bool(temporary_cache),
            "source": SourceIdentity.from_path(
                source_path,
                source_mtime_ns,
                source_size,
            ).to_json(),
        }
        if source_paths is not None:
            metadata["source_files"] = [
                SourceIdentity.from_path(path, path.stat().st_mtime_ns, path.stat().st_size).to_json()
                for path in source_paths
            ]
        if cells is not None:
            metadata["cell_shape"] = [int(frame_count), 3, 3]
        if frame_availability is not None:
            metadata["frame_availability_file"] = FRAME_AVAILABILITY_FILE
        if symbols is not None:
            metadata["unique_symbols"] = sorted(set(symbols))
        _write_metadata(root, metadata)

        atom_numbers_map = np.memmap(
            root / ATOM_NUMBERS_FILE,
            dtype=np.uint16,
            mode="r",
            shape=(atom_count,),
        )
        return cls(
            root,
            metadata,
            positions,
            atom_numbers_map,
            cells,
            frame_availability,
        )

    @classmethod
    def open(cls, root: Path, mode: str = "r") -> "BinaryTrajectoryStore":
        root = root.resolve()
        if mode not in {"r", "r+", "c"}:
            raise ValueError(f"Unsupported trajectory store mode: {mode}")

        metadata_path = _validated_member(root, METADATA_FILE)
        metadata_size = metadata_path.stat().st_size
        if metadata_size > MAX_METADATA_BYTES:
            raise CacheValidationError(
                f"Trajectory metadata is too large: {metadata_size} bytes"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CacheValidationError(f"Invalid trajectory metadata: {exc}") from exc
        if not isinstance(metadata, dict):
            raise CacheValidationError("Trajectory metadata must be a JSON object")

        version = _metadata_int(metadata, "version", minimum=0)
        if version not in SUPPORTED_STORE_VERSIONS:
            raise CacheValidationError(
                f"Unsupported trajectory store version: {metadata.get('version')}"
            )
        if metadata.get("dtype") != "float32":
            raise CacheValidationError(
                f"Unsupported trajectory dtype: {metadata.get('dtype')}"
            )

        frame_count = _metadata_int(metadata, "frame_count", minimum=1)
        atom_count = _metadata_int(metadata, "atom_count", minimum=1)
        expected_shape = (frame_count, atom_count, 3)
        shape = metadata.get("shape")
        if shape is not None and not _shape_matches(shape, expected_shape):
            raise CacheValidationError(
                f"Invalid position shape {shape!r}; expected {list(expected_shape)!r}"
            )

        available_frame_count = _metadata_int(
            metadata,
            "available_frame_count",
            minimum=0,
            default=frame_count if metadata.get("complete", True) else 0,
        )
        if available_frame_count > frame_count:
            raise CacheValidationError(
                "available_frame_count exceeds frame_count"
            )
        _validate_metadata_structure(metadata, atom_count=atom_count)

        positions_path = _validated_sized_member(
            root,
            POSITIONS_FILE,
            _array_nbytes("positions", expected_shape, np.dtype(np.float32).itemsize),
        )
        atom_numbers_path = _validated_sized_member(
            root,
            ATOM_NUMBERS_FILE,
            _array_nbytes("atom numbers", (atom_count,), np.dtype(np.uint16).itemsize),
        )

        cell_shape = metadata.get("cell_shape")
        cells_path: Path | None = None
        if cell_shape is not None:
            expected_cell_shape = (frame_count, 3, 3)
            if not _shape_matches(cell_shape, expected_cell_shape):
                raise CacheValidationError(
                    f"Invalid cell shape {cell_shape!r}; expected {list(expected_cell_shape)!r}"
                )
            cells_path = _validated_sized_member(
                root,
                CELLS_FILE,
                _array_nbytes("cells", expected_cell_shape, np.dtype(np.float32).itemsize),
            )

        availability_name = metadata.get("frame_availability_file")
        availability_path: Path | None = None
        if availability_name is not None:
            if availability_name != FRAME_AVAILABILITY_FILE:
                raise CacheValidationError(
                    "Invalid frame availability member name"
                )
            availability_path = _validated_sized_member(
                root,
                FRAME_AVAILABILITY_FILE,
                _array_nbytes("frame availability", (frame_count,), np.dtype(np.uint8).itemsize),
            )

        positions = np.memmap(
            positions_path,
            dtype=np.float32,
            mode=mode,
            shape=expected_shape,
        )
        cells = None
        if cells_path is not None:
            cells = np.memmap(
                cells_path,
                dtype=np.float32,
                mode=mode,
                shape=(frame_count, 3, 3),
            )
        atom_numbers = np.memmap(
            atom_numbers_path,
            dtype=np.uint16,
            mode="r",
            shape=(atom_count,),
        )
        frame_availability = None
        if availability_path is not None:
            frame_availability = np.memmap(
                availability_path,
                dtype=np.uint8,
                mode=mode,
                shape=(frame_count,),
            )
        return cls(root, metadata, positions, atom_numbers, cells, frame_availability)

    def frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        return self.positions[frame_index]

    def cell(self, frame_index: int) -> np.ndarray | None:
        if self.cells is None:
            return None
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        return self.cells[frame_index]

    def is_frame_available(self, frame_index: int) -> bool:
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            return False
        with self._availability_lock:
            if self.frame_availability is not None:
                return bool(self.frame_availability[index])
            return index < self._available_frame_count

    def mark_frame_available(self, available_frame_count: int) -> None:
        count = max(0, min(int(available_frame_count), self.frame_count))
        with self._availability_lock:
            if self.frame_availability is not None:
                start = self._available_prefix_count
                if count <= start:
                    return
                missing = (count - start) - int(
                    np.count_nonzero(self.frame_availability[start:count])
                )
                self.frame_availability[start:count] = 1
                self._available_frame_count += missing
                self._available_prefix_count = count
            elif count > self._available_frame_count:
                self._available_frame_count = count
                self._available_prefix_count = count

    def publish_frame(self, frame_index: int) -> None:
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        with self._availability_lock:
            if self.frame_availability is None:
                if index != self._available_prefix_count:
                    raise RuntimeError("This cache only supports sequential frame publication")
                self._available_frame_count += 1
                self._available_prefix_count += 1
                return
            if self.frame_availability[index]:
                return
            self.frame_availability[index] = 1
            self._available_frame_count += 1
            if index == self._available_prefix_count:
                prefix = index + 1
                while prefix < self.frame_count and self.frame_availability[prefix]:
                    prefix += 1
                self._available_prefix_count = prefix

    def mark_complete(self) -> None:
        with self._availability_lock:
            if self._available_frame_count < self.frame_count:
                raise RuntimeError("Cannot complete a trajectory cache before every frame is available")
            self.metadata["complete"] = True
            self.metadata["available_frame_count"] = self.frame_count
        self.flush()
        _write_metadata(self.root, self.metadata)

    def flush(self) -> None:
        self.positions.flush()
        if self.cells is not None:
            self.cells.flush()
        if self.frame_availability is not None:
            self.frame_availability.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        arrays = [self.positions, self.atom_numbers]
        if self.cells is not None:
            arrays.append(self.cells)
        if self.frame_availability is not None:
            arrays.append(self.frame_availability)
        for array in arrays:
            mmap_obj = getattr(array, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        self._closed = True
        if self.metadata.get("temporary_cache"):
            try:
                shutil.rmtree(self.root)
            except OSError:
                pass

    def __enter__(self) -> "BinaryTrajectoryStore":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def is_valid_for_source(self, source_path: Path) -> bool:
        if not source_path.exists():
            return False
        stat = source_path.stat()
        expected = SourceIdentity.from_path(source_path, stat.st_mtime_ns, stat.st_size)
        return (
            self.source.path == expected.path
            and self.source.mtime_ns == expected.mtime_ns
            and self.source.size == expected.size
        )

    def is_valid_for_sources(self, source_paths: Sequence[Path]) -> bool:
        paths = tuple(Path(path).resolve() for path in source_paths)
        stored = self.metadata.get("source_files")
        if stored is None:
            return len(paths) == 1 and self.is_valid_for_source(paths[0])
        if len(stored) != len(paths):
            return False
        for data, path in zip(stored, paths):
            if not path.exists():
                return False
            stat = path.stat()
            expected = SourceIdentity.from_path(path, stat.st_mtime_ns, stat.st_size)
            if SourceIdentity.from_json(data) != expected:
                return False
        return True


def _write_metadata(root: Path, metadata: dict[str, Any]) -> None:
    target = root / METADATA_FILE
    temporary = root / f"{METADATA_FILE}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(target)


def _metadata_int(
    metadata: dict[str, Any],
    key: str,
    *,
    minimum: int,
    default: int | None = None,
) -> int:
    value = metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CacheValidationError(f"Trajectory metadata {key!r} must be an integer")
    if value < minimum:
        raise CacheValidationError(
            f"Trajectory metadata {key!r} must be at least {minimum}"
        )
    return value


def _shape_matches(value: object, expected: tuple[int, ...]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    return all(
        not isinstance(item, bool) and isinstance(item, int) and item == wanted
        for item, wanted in zip(value, expected)
    )


def _validate_metadata_structure(metadata: dict[str, Any], *, atom_count: int) -> None:
    for key in ("complete", "random_access", "temporary_cache"):
        if key in metadata and not isinstance(metadata[key], bool):
            raise CacheValidationError(f"Trajectory metadata {key!r} must be boolean")

    source = metadata.get("source", {})
    _validate_source_identity(source, field="source")
    source_files = metadata.get("source_files")
    if source_files is not None:
        if not isinstance(source_files, list) or not source_files:
            raise CacheValidationError("Trajectory metadata 'source_files' must be a non-empty list")
        for index, identity in enumerate(source_files):
            _validate_source_identity(identity, field=f"source_files[{index}]")

    symbols = metadata.get("unique_symbols")
    if symbols is not None and (
        not isinstance(symbols, list)
        or len(symbols) > atom_count
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise CacheValidationError("Trajectory metadata 'unique_symbols' is invalid")


def _validate_source_identity(value: object, *, field: str) -> None:
    if not isinstance(value, dict):
        raise CacheValidationError(f"Trajectory metadata {field!r} must be an object")
    path = value.get("path")
    if path is not None and not isinstance(path, str):
        raise CacheValidationError(f"Trajectory metadata {field!r} has an invalid path")
    for key in ("mtime_ns", "size"):
        number = value.get(key, 0)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise CacheValidationError(
                f"Trajectory metadata {field!r} has an invalid {key}"
            )


def _array_nbytes(name: str, shape: tuple[int, ...], itemsize: int) -> int:
    size = int(itemsize)
    for dimension in shape:
        if dimension <= 0 or size > sys.maxsize // dimension:
            raise CacheValidationError(f"Trajectory {name} dimensions are too large")
        size *= dimension
    return size


def _validated_member(root: Path, name: str) -> Path:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise CacheValidationError(f"Invalid trajectory cache member: {name!r}")
    try:
        member = (root / name).resolve(strict=True)
    except OSError as exc:
        raise CacheValidationError(
            f"Missing trajectory cache member: {name}"
        ) from exc
    if not member.is_relative_to(root) or not member.is_file():
        raise CacheValidationError(
            f"Trajectory cache member escapes its cache directory: {name}"
        )
    return member


def _validated_sized_member(root: Path, name: str, expected_bytes: int) -> Path:
    member = _validated_member(root, name)
    actual_bytes = member.stat().st_size
    if actual_bytes != expected_bytes:
        raise CacheValidationError(
            f"Invalid {name} size: {actual_bytes} bytes; expected {expected_bytes}"
        )
    return member
