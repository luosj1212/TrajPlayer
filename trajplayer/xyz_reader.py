from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from .reader_common import (
    MAX_ATOMIC_NUMBER,
    StructureFrame,
    normalize_symbol,
    numbers_to_symbols,
    require_finite_coordinates,
    symbols_to_numbers,
    validate_text_atom_count,
)


_MINIMUM_XYZ_ATOM_BYTES = 7
_FLOAT32_MAX = float(np.finfo(np.float32).max)


_KEY_VALUE_PATTERN = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>\S+))"
)


@dataclass(frozen=True, slots=True)
class XyzFrameLayout:
    data_offset: int
    identity_column: int
    identity_is_atomic_number: bool
    position_columns: tuple[int, int, int]
    expected_columns: int
    cell: np.ndarray | None


def inspect_xyz_frame_buffer(
    source,
    frame_offset: int,
    atom_count: int,
    frame_index: int,
) -> XyzFrameLayout:
    """Inspect one XYZ header/comment without decoding its atom rows."""

    start = int(frame_offset)
    source_size = len(source)
    if start < 0 or start >= source_size:
        raise ValueError(f"XYZ frame {frame_index} ended before its atom count")
    header_end = source.find(b"\n", start, source_size)
    if header_end < 0:
        raise ValueError(f"XYZ frame {frame_index} ended before its atom count")
    try:
        parsed_atom_count = int(bytes(source[start:header_end]).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ frame header at frame {frame_index}") from exc
    if parsed_atom_count != int(atom_count):
        raise ValueError(
            f"Frame {frame_index} has {parsed_atom_count} atoms; expected {atom_count}"
        )

    comment_start = header_end + 1
    comment_end = source.find(b"\n", comment_start, source_size)
    if comment_end < 0:
        raise ValueError(f"XYZ frame {frame_index} is missing its comment line")
    comment = bytes(source[comment_start:comment_end]).decode(
        "utf-8",
        errors="replace",
    )
    validate_text_atom_count(
        atom_count,
        available_bytes=source_size - (comment_end + 1),
        minimum_atom_bytes=_MINIMUM_XYZ_ATOM_BYTES,
        format_name=f"XYZ frame {frame_index}",
    )
    metadata = _parse_comment(comment.strip())
    properties = _parse_properties(metadata.get("Properties"))
    species_column, number_column, position_columns, expected_columns = _column_layout(
        properties
    )
    return XyzFrameLayout(
        data_offset=comment_end + 1,
        identity_column=(
            int(number_column) if number_column is not None else int(species_column)
        ),
        identity_is_atomic_number=number_column is not None,
        position_columns=position_columns,
        expected_columns=expected_columns,
        cell=_parse_lattice(metadata.get("Lattice")),
    )


def read_xyz_frame(handle: BinaryIO, atom_count: int, frame_index: int) -> StructureFrame:
    header = handle.readline()
    if not header:
        raise ValueError(f"XYZ frame {frame_index} ended before its atom count")
    try:
        parsed_atom_count = int(header.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ frame header at frame {frame_index}") from exc
    if parsed_atom_count != int(atom_count):
        raise ValueError(
            f"Frame {frame_index} has {parsed_atom_count} atoms; expected {atom_count}"
        )
    comment_line = handle.readline()
    if not comment_line:
        raise ValueError(f"XYZ frame {frame_index} is missing its comment line")
    metadata = _parse_comment(comment_line.decode("utf-8", errors="replace").strip())
    properties = _parse_properties(metadata.get("Properties"))
    species_column, number_column, position_columns, expected_columns = _column_layout(properties)

    remaining_bytes = _remaining_binary_bytes(handle)
    if remaining_bytes is not None:
        validate_text_atom_count(
            atom_count,
            available_bytes=remaining_bytes,
            minimum_atom_bytes=_MINIMUM_XYZ_ATOM_BYTES,
            format_name=f"XYZ frame {frame_index}",
        )

    positions = np.empty((atom_count, 3), dtype=np.float32)
    symbols: list[str] = []
    numbers = np.empty(atom_count, dtype=np.uint16) if number_column is not None else None
    for atom_index in range(atom_count):
        line = handle.readline()
        if not line:
            raise ValueError(f"XYZ frame {frame_index} ended at atom {atom_index}")
        values = line.decode("utf-8", errors="replace").split()
        if len(values) < expected_columns:
            raise ValueError(
                f"XYZ frame {frame_index}, atom {atom_index} has {len(values)} columns; "
                f"expected at least {expected_columns}"
            )
        try:
            coordinates = [float(values[column]) for column in position_columns]
            if not all(
                math.isfinite(value) and abs(value) <= _FLOAT32_MAX
                for value in coordinates
            ):
                raise ValueError("coordinates must be finite float32 values")
            positions[atom_index] = coordinates
            if number_column is not None and numbers is not None:
                atomic_number = int(values[number_column])
                if atomic_number < 0 or atomic_number > MAX_ATOMIC_NUMBER:
                    raise ValueError("atomic number is out of range")
                numbers[atom_index] = atomic_number
            else:
                symbols.append(normalize_symbol(values[species_column]))
        except (TypeError, ValueError, OverflowError, IndexError) as exc:
            raise ValueError(
                f"Invalid XYZ data at frame {frame_index}, atom {atom_index}"
            ) from exc

    if numbers is None:
        atom_numbers = symbols_to_numbers(symbols)
        normalized_symbols = tuple(symbols)
    else:
        atom_numbers = np.ascontiguousarray(numbers, dtype=np.uint16)
        normalized_symbols = numbers_to_symbols(atom_numbers)
    cell = _parse_lattice(metadata.get("Lattice"))
    return StructureFrame(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        atom_numbers=atom_numbers,
        symbols=normalized_symbols,
        cell=cell,
    )


def _parse_comment(comment: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _KEY_VALUE_PATTERN.finditer(comment):
        value = match.group("double")
        if value is None:
            value = match.group("single")
        if value is None:
            value = match.group("bare") or ""
        result[match.group("key")] = value
    return result


def _parse_properties(value: str | None) -> list[tuple[str, str, int]]:
    if not value:
        return [("species", "S", 1), ("pos", "R", 3)]
    fields = value.split(":")
    if len(fields) % 3:
        raise ValueError("XYZ Properties metadata must contain name:type:count triplets")
    properties: list[tuple[str, str, int]] = []
    for index in range(0, len(fields), 3):
        try:
            count = int(fields[index + 2])
        except ValueError as exc:
            raise ValueError("XYZ property column count is invalid") from exc
        if count <= 0:
            raise ValueError("XYZ property column count must be positive")
        properties.append((fields[index], fields[index + 1], count))
    return properties


def _column_layout(
    properties: list[tuple[str, str, int]],
) -> tuple[int, int | None, tuple[int, int, int], int]:
    species_column: int | None = None
    number_column: int | None = None
    position_columns: tuple[int, int, int] | None = None
    column = 0
    for name, _kind, count in properties:
        normalized = name.casefold()
        if normalized in {"species", "symbol", "symbols"} and count == 1:
            species_column = column
        elif normalized in {"z", "numbers", "atomic_number", "atomic_numbers"} and count == 1:
            number_column = column
        elif normalized in {"pos", "position", "positions"} and count == 3:
            position_columns = (column, column + 1, column + 2)
        column += count
    if position_columns is None:
        raise ValueError("XYZ frame does not define a three-column position property")
    if species_column is None and number_column is None:
        raise ValueError("XYZ frame does not define atomic species or atomic numbers")
    return species_column or 0, number_column, position_columns, column


def _parse_lattice(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    try:
        values = [float(token) for token in value.replace(",", " ").split()]
    except ValueError as exc:
        raise ValueError("XYZ Lattice metadata contains a non-numeric value") from exc
    if len(values) != 9:
        raise ValueError("XYZ Lattice metadata must contain nine values")
    with np.errstate(over="ignore", invalid="ignore"):
        cell = np.asarray(values, dtype=np.float32).reshape(3, 3)
    require_finite_coordinates(cell, context="XYZ Lattice metadata")
    return np.ascontiguousarray(cell, dtype=np.float32) if np.any(cell) else None


def _remaining_binary_bytes(handle: BinaryIO) -> int | None:
    try:
        current = int(handle.tell())
    except (AttributeError, OSError, ValueError):
        return None
    try:
        return max(0, int(os.fstat(handle.fileno()).st_size) - current)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        handle.seek(0, os.SEEK_END)
        end = int(handle.tell())
        handle.seek(current, os.SEEK_SET)
    except (AttributeError, OSError, ValueError):
        return None
    return max(0, end - current)
