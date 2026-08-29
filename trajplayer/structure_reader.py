from __future__ import annotations

import math
import re
import shlex
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

from .reader_common import (
    StructureFrame,
    cell_from_lengths_angles,
    first_alpha_run,
    normalize_symbol,
    symbols_to_numbers,
)


def read_structure(path: Path) -> StructureFrame:
    suffix = path.suffix.lower()
    if suffix not in {".gro", ".pdb", ".cif"}:
        raise ValueError(f"Unsupported structure format: {path.name}")
    try:
        return _read_chemfiles_structure(path)
    except ImportError:
        if suffix == ".gro":
            return read_gro(path)
        if suffix == ".pdb":
            return read_pdb(path)
        return read_cif(path)


def read_gro(path: Path) -> StructureFrame:
    source = path.resolve()
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        if not handle.readline():
            raise ValueError("GRO file is empty")
        count_line = handle.readline()
        try:
            atom_count = int(count_line.strip())
        except ValueError as exc:
            raise ValueError("GRO atom-count line is invalid") from exc
        if atom_count <= 0:
            raise ValueError("GRO atom count must be positive")

        positions = np.empty((atom_count, 3), dtype=np.float32)
        symbols: list[str] = []
        for atom_index in range(atom_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"GRO file ended at atom {atom_index}")
            if len(line) < 44:
                raise ValueError(f"GRO atom line {atom_index + 3} is too short")
            residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            try:
                positions[atom_index] = [
                    float(line[20:28]) * 10.0,
                    float(line[28:36]) * 10.0,
                    float(line[36:44]) * 10.0,
                ]
            except ValueError as exc:
                raise ValueError(f"GRO coordinates are invalid at atom {atom_index}") from exc
            symbols.append(_gro_symbol(atom_name, residue_name))

        box_line = handle.readline()
    cell = _gro_cell(box_line)
    return StructureFrame(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        atom_numbers=symbols_to_numbers(symbols),
        symbols=tuple(symbols),
        cell=cell,
    )


def read_pdb(path: Path) -> StructureFrame:
    positions: list[tuple[float, float, float]] = []
    symbols: list[str] = []
    cell: np.ndarray | None = None
    in_first_model = False
    saw_model = False
    with path.resolve().open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = line[:6].strip().upper()
            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if record == "CRYST1":
                try:
                    cell = cell_from_lengths_angles(
                        float(line[6:15]),
                        float(line[15:24]),
                        float(line[24:33]),
                        float(line[33:40]),
                        float(line[40:47]),
                        float(line[47:54]),
                    )
                except ValueError as exc:
                    raise ValueError(f"PDB CRYST1 record is invalid at line {line_number}") from exc
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            if saw_model and not in_first_model:
                continue
            try:
                positions.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            except ValueError as exc:
                raise ValueError(f"PDB coordinates are invalid at line {line_number}") from exc
            element_field = line[76:78].strip() if len(line) >= 78 else ""
            atom_field = line[12:16] if len(line) >= 16 else ""
            symbols.append(_pdb_symbol(element_field, atom_field))
    if not positions:
        raise ValueError("No atoms found in PDB file")
    position_array = np.ascontiguousarray(positions, dtype=np.float32)
    return StructureFrame(
        positions=position_array,
        atom_numbers=symbols_to_numbers(symbols),
        symbols=tuple(symbols),
        cell=cell,
    )


def read_cif(path: Path) -> StructureFrame:
    scalars, loops = _read_cif_tables(path)
    cell = _cif_cell(scalars)
    atom_headers: list[str] | None = None
    atom_rows: list[list[str]] | None = None
    for headers, rows in loops:
        lowered = {header.casefold() for header in headers}
        if any(header.startswith("_atom_site_") for header in lowered):
            atom_headers = headers
            atom_rows = rows
            break
    if atom_headers is None or atom_rows is None:
        raise ValueError("CIF file has no atom_site loop")

    lookup = {name.casefold(): index for index, name in enumerate(atom_headers)}
    symbol_index = _first_index(
        lookup,
        "_atom_site_type_symbol",
        "_atom_site_label",
    )
    cartesian_indices = _coordinate_indices(lookup, "cartn")
    fractional_indices = _coordinate_indices(lookup, "fract")
    if cartesian_indices is None and fractional_indices is None:
        raise ValueError("CIF atom_site loop has no Cartesian or fractional coordinates")
    if fractional_indices is not None and cell is None:
        raise ValueError("CIF fractional coordinates require complete cell parameters")

    positions = np.empty((len(atom_rows), 3), dtype=np.float32)
    symbols: list[str] = []
    for row_index, row in enumerate(atom_rows):
        if len(row) != len(atom_headers):
            raise ValueError(f"CIF atom row {row_index} has the wrong number of columns")
        symbols.append(normalize_symbol(row[symbol_index]))
        indices = cartesian_indices or fractional_indices
        assert indices is not None
        coordinates = np.asarray(
            [_cif_number(row[index]) for index in indices], dtype=np.float64
        )
        if fractional_indices is not None and cartesian_indices is None:
            assert cell is not None
            coordinates = coordinates @ np.asarray(cell, dtype=np.float64)
        positions[row_index] = coordinates
    if positions.shape[0] == 0:
        raise ValueError("No atoms found in CIF file")
    return StructureFrame(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        atom_numbers=symbols_to_numbers(symbols),
        symbols=tuple(symbols),
        cell=cell,
    )


def _read_chemfiles_structure(path: Path) -> StructureFrame:
    from chemfiles import Trajectory

    trajectory = Trajectory(str(path.resolve()), mode="r")
    try:
        if int(trajectory.nsteps) <= 0:
            raise ValueError(f"No frames found in {path.name}")
        frame = trajectory.read_step(0)
        positions = np.ascontiguousarray(frame.positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] <= 0:
            raise ValueError(f"Invalid coordinate array in {path.name}")
        numbers = np.empty(positions.shape[0], dtype=np.uint16)
        symbols: list[str] = []
        atoms = frame.topology.atoms
        for atom_index in range(positions.shape[0]):
            atom = atoms[atom_index]
            number = int(atom.atomic_number)
            identity = atom.type or atom.name
            if path.suffix.lower() == ".pdb" and atom.name[:1].isdigit():
                identity = first_alpha_run(atom.name)[:1]
            symbol = normalize_symbol(identity)
            if number <= 0:
                number = int(atomic_numbers.get(symbol, 0))
            numbers[atom_index] = number
            if number > 0:
                symbol = chemical_symbols[number]
            symbols.append(symbol)
        cell = np.asarray(frame.cell.matrix, dtype=np.float32).T
        cell_value = np.ascontiguousarray(cell, dtype=np.float32) if np.any(cell) else None
        return StructureFrame(
            positions=positions,
            atom_numbers=np.ascontiguousarray(numbers, dtype=np.uint16),
            symbols=tuple(symbols),
            cell=cell_value,
        )
    finally:
        trajectory.close()


def _gro_symbol(atom_name: str, residue_name: str) -> str:
    label = first_alpha_run(atom_name)
    if not label:
        return "X"
    exact = label[:2]
    if exact in atomic_numbers:
        return exact
    residue_symbol = normalize_symbol(residue_name)
    normalized_pair = normalize_symbol(exact)
    if len(exact) >= 2 and residue_symbol == normalized_pair:
        return normalized_pair
    first = label[0].upper()
    if first in atomic_numbers:
        return first
    last = label[-1].upper()
    return last if last in atomic_numbers else "X"


def _gro_cell(line: str) -> np.ndarray | None:
    if not line.strip():
        return None
    try:
        values = [float(value) for value in line.split()]
    except ValueError as exc:
        raise ValueError("GRO box record is invalid") from exc
    if len(values) < 3:
        return None
    cell = np.diag(values[:3]).astype(np.float32)
    if len(values) >= 9:
        cell.flat[[1, 2, 3, 5, 6, 7]] = values[3:9]
    cell *= np.float32(10.0)
    return np.ascontiguousarray(cell, dtype=np.float32) if np.any(cell) else None


def _pdb_symbol(element_field: str, atom_field: str) -> str:
    if element_field:
        return normalize_symbol(element_field)
    letters = first_alpha_run(atom_field)
    if not letters:
        return "X"
    if atom_field and (atom_field[0].isspace() or atom_field[0].isdigit()):
        return normalize_symbol(letters[0])
    return normalize_symbol(atom_field)


def _read_cif_tables(
    path: Path,
) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    lines = path.resolve().read_text(encoding="utf-8", errors="replace").splitlines()
    scalars: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines):
        tokens = _cif_tokens(lines[index])
        if not tokens:
            index += 1
            continue
        first = tokens[0].casefold()
        if first == "loop_":
            index += 1
            headers: list[str] = []
            while index < len(lines):
                header_tokens = _cif_tokens(lines[index])
                if not header_tokens:
                    index += 1
                    continue
                if not header_tokens[0].startswith("_"):
                    break
                headers.append(header_tokens[0])
                index += 1
            values: list[str] = []
            while index < len(lines):
                row_tokens = _cif_tokens(lines[index])
                if not row_tokens:
                    index += 1
                    continue
                marker = row_tokens[0].casefold()
                if marker == "loop_" or marker.startswith("data_") or marker == "stop_":
                    break
                if row_tokens[0].startswith("_") and len(values) % max(1, len(headers)) == 0:
                    break
                values.extend(row_tokens)
                index += 1
            if not headers:
                raise ValueError("CIF loop has no headers")
            if len(values) % len(headers):
                raise ValueError("CIF loop has an incomplete row")
            rows = [values[start : start + len(headers)] for start in range(0, len(values), len(headers))]
            loops.append((headers, rows))
            continue
        if tokens[0].startswith("_"):
            if len(tokens) >= 2:
                scalars[tokens[0].casefold()] = tokens[1]
                index += 1
                continue
            index += 1
            while index < len(lines):
                value_tokens = _cif_tokens(lines[index])
                if value_tokens:
                    scalars[tokens[0].casefold()] = value_tokens[0]
                    index += 1
                    break
                index += 1
            continue
        index += 1
    return scalars, loops


def _cif_tokens(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    try:
        return shlex.split(stripped, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError("CIF contains an unterminated quoted value") from exc


def _cif_number(value: str) -> float:
    if value in {".", "?"}:
        raise ValueError("CIF coordinate is missing")
    cleaned = re.sub(r"\([^)]*\)$", "", value)
    try:
        result = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid CIF numeric value: {value}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite CIF numeric value: {value}")
    return result


def _cif_cell(scalars: dict[str, str]) -> np.ndarray | None:
    length_keys = ("_cell_length_a", "_cell_length_b", "_cell_length_c")
    if not all(key in scalars for key in length_keys):
        return None
    lengths = [_cif_number(scalars[key]) for key in length_keys]
    angles = [
        _cif_number(scalars.get(key, "90"))
        for key in ("_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
    ]
    return cell_from_lengths_angles(*lengths, *angles)


def _first_index(lookup: dict[str, int], *names: str) -> int:
    for name in names:
        if name in lookup:
            return lookup[name]
    raise ValueError(f"CIF atom_site loop is missing {names[0]}")


def _coordinate_indices(
    lookup: dict[str, int], coordinate_kind: str
) -> tuple[int, int, int] | None:
    names = tuple(f"_atom_site_{coordinate_kind}_{axis}" for axis in "xyz")
    if not all(name in lookup for name in names):
        return None
    return tuple(lookup[name] for name in names)
