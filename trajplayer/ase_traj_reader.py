from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .reader_common import numbers_to_symbols


class AseUlmTrajectoryReader:
    """Read the array subset of ASE's documented ULM trajectory format."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._file = self.path.open("rb")
        self._offsets: np.memmap | None = None
        try:
            if self._file.read(8) not in {b"- of Ulm", b"AFFormat"}:
                raise ValueError("Not an ASE ULM trajectory")
            tag = self._file.read(16).decode("ascii", errors="strict").rstrip()
            if tag != "ASE-Trajectory":
                raise ValueError(f"Unsupported ULM tag: {tag or '<empty>'}")
            header = np.frombuffer(self._file.read(24), dtype="<i8", count=3)
            if header.size != 3:
                raise ValueError("ASE trajectory header is truncated")
            version, frame_count, offsets_position = (int(value) for value in header)
            if version < 1 or version > 3:
                raise ValueError(f"Unsupported ASE ULM version: {version}")
            if frame_count <= 0:
                raise ValueError("No frames found in trajectory")
            self._file_size = self.path.stat().st_size
            if offsets_position < 48 or offsets_position + frame_count * 8 > self._file_size:
                raise ValueError("ASE trajectory offset table is invalid")
            self.frame_count = frame_count
            self._offsets = np.memmap(
                self.path,
                mode="r",
                dtype="<i8",
                offset=offsets_position,
                shape=(frame_count,),
            )
            first = self._read_item(0)
            self.atom_numbers = self._read_required_array(first, "numbers", 0, np.uint16)
            if self.atom_numbers.ndim != 1 or self.atom_numbers.size <= 0:
                raise ValueError("ASE trajectory has an invalid atomic-number array")
            self.atom_numbers = np.ascontiguousarray(self.atom_numbers, dtype=np.uint16)
            self.symbols = numbers_to_symbols(self.atom_numbers)
            first_positions = self._positions_from_item(first, 0)
            if first_positions.shape != (self.atom_numbers.size, 3):
                raise ValueError(
                    f"Frame 0 has position shape {first_positions.shape}; expected "
                    f"{(self.atom_numbers.size, 3)}"
                )
            self._default_cell = self._cell_from_item(first)
            self.has_cell = self._default_cell is not None
            self._prefetched_first_frame = (first_positions, self._default_cell)
        except Exception:
            self.close()
            raise

    @property
    def atom_count(self) -> int:
        return int(self.atom_numbers.size)

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        prefetched = getattr(self, "_prefetched_first_frame", None)
        if index == 0 and prefetched is not None:
            self._prefetched_first_frame = None
            return prefetched
        item = self._read_item(index)
        positions = self._positions_from_item(item, index)
        if positions.shape != (self.atom_count, 3):
            raise ValueError(
                f"Frame {index} has position shape {positions.shape}; expected "
                f"{(self.atom_count, 3)}"
            )
        if "numbers." in item:
            numbers = self._read_required_array(item, "numbers", index, np.uint16)
            if not np.array_equal(numbers, self.atom_numbers):
                raise ValueError(f"Frame {index} atom ordering differs from the first frame")
        cell = self._cell_from_item(item)
        if cell is None and "cell" not in item:
            cell = self._default_cell
        return positions, cell

    def close(self) -> None:
        self._prefetched_first_frame = None
        offsets = getattr(self, "_offsets", None)
        if offsets is not None:
            mmap = getattr(offsets, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self._offsets = None
        handle = getattr(self, "_file", None)
        if handle is not None:
            handle.close()
            self._file = None

    def _read_item(self, frame_index: int) -> dict[str, object]:
        if self._offsets is None or self._file is None:
            raise RuntimeError("ASE trajectory reader is closed")
        offset = int(self._offsets[int(frame_index)])
        if offset < 0:
            raise ValueError(f"Frame {frame_index} has an invalid ULM offset")
        self._file.seek(offset)
        size_buffer = self._file.read(8)
        if len(size_buffer) != 8:
            raise ValueError(f"Frame {frame_index} metadata is truncated")
        metadata_size = int(np.frombuffer(size_buffer, dtype="<i8", count=1)[0])
        if metadata_size < 0 or metadata_size > self._file_size:
            raise ValueError(f"Frame {frame_index} metadata size is invalid")
        metadata = self._file.read(metadata_size)
        if len(metadata) != metadata_size:
            raise ValueError(f"Frame {frame_index} metadata is truncated")
        value = json.loads(metadata.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Frame {frame_index} metadata is not an object")
        return value

    def _positions_from_item(self, item: dict[str, object], frame_index: int) -> np.ndarray:
        return self._read_required_array(item, "positions", frame_index, np.float32)

    def _read_required_array(
        self,
        item: dict[str, object],
        name: str,
        frame_index: int,
        output_dtype: np.dtype,
    ) -> np.ndarray:
        descriptor = item.get(f"{name}.")
        if not isinstance(descriptor, dict) or "ndarray" not in descriptor:
            raise ValueError(f"Frame {frame_index} is missing its {name} array")
        payload = descriptor["ndarray"]
        if not isinstance(payload, (list, tuple)) or len(payload) != 3:
            raise ValueError(f"Frame {frame_index} has an invalid {name} descriptor")
        shape_value, dtype_value, offset_value = payload
        shape = tuple(int(size) for size in shape_value)
        if not shape or any(size < 0 for size in shape):
            raise ValueError(f"Frame {frame_index} has an invalid {name} shape")
        dtype = np.dtype(str(dtype_value))
        little_endian = item.get("_little_endian", True) is not False
        dtype = dtype.newbyteorder("<" if little_endian else ">")
        count = int(np.prod(shape, dtype=np.int64))
        offset = int(offset_value)
        if offset < 0 or offset + count * dtype.itemsize > self._file_size:
            raise ValueError(f"Frame {frame_index} {name} array lies outside the file")
        if self._file is None:
            raise RuntimeError("ASE trajectory reader is closed")
        self._file.seek(offset)
        raw = self._file.read(count * dtype.itemsize)
        if len(raw) != count * dtype.itemsize:
            raise ValueError(f"Frame {frame_index} {name} array is truncated")
        array = np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)
        return np.ascontiguousarray(array, dtype=output_dtype)

    @staticmethod
    def _cell_from_item(item: dict[str, object]) -> np.ndarray | None:
        value = item.get("cell")
        if value is None:
            return None
        cell = np.asarray(value, dtype=np.float32)
        if cell.shape != (3, 3):
            raise ValueError(f"ASE trajectory cell has shape {cell.shape}; expected (3, 3)")
        if not np.any(cell):
            return None
        return np.ascontiguousarray(cell, dtype=np.float32)
