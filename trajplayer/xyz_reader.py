from __future__ import annotations

import re
from typing import BinaryIO

import numpy as np

from .reader_common import StructureFrame, normalize_symbol, numbers_to_symbols, symbols_to_numbers


_KEY_VALUE_PATTERN = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>\S+))"
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
            positions[atom_index] = [float(values[column]) for column in position_columns]
            if number_column is not None and numbers is not None:
                numbers[atom_index] = int(values[number_column])
            else:
                symbols.append(normalize_symbol(values[species_column]))
        except (TypeError, ValueError, IndexError) as exc:
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
    cell = np.asarray(values, dtype=np.float32).reshape(3, 3)
    return np.ascontiguousarray(cell, dtype=np.float32) if np.any(cell) else None
