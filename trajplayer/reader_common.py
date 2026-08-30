from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ase.data import atomic_numbers, chemical_symbols


MAX_ATOMIC_NUMBER = len(chemical_symbols) - 1


@dataclass(frozen=True)
class StructureFrame:
    positions: np.ndarray
    atom_numbers: np.ndarray
    symbols: tuple[str, ...]
    cell: np.ndarray | None


def first_alpha_run(value: str) -> str:
    """Return the first contiguous ASCII-letter run in an atom label."""

    run: list[str] = []
    for character in str(value):
        is_ascii_letter = "A" <= character <= "Z" or "a" <= character <= "z"
        if is_ascii_letter:
            run.append(character)
        elif run:
            break
    return "".join(run)


def normalize_symbol(value: str) -> str:
    letters = first_alpha_run(value)
    if not letters:
        return "X"
    for width in (2, 1):
        candidate = letters[:width].capitalize()
        if candidate in atomic_numbers:
            return candidate
    return "X"


def symbols_to_numbers(symbols: list[str] | tuple[str, ...]) -> np.ndarray:
    return np.ascontiguousarray(
        [atomic_numbers.get(normalize_symbol(symbol), 0) for symbol in symbols],
        dtype=np.uint16,
    )


def numbers_to_symbols(numbers: np.ndarray) -> tuple[str, ...]:
    values = np.asarray(numbers, dtype=np.int64)
    return tuple(
        chemical_symbols[number] if 0 <= number < len(chemical_symbols) else "X"
        for number in values
    )


def validated_atom_numbers(numbers: np.ndarray, *, context: str) -> np.ndarray:
    values = np.asarray(numbers)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError(f"{context} must be a non-empty one-dimensional array")
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{context} must contain integer atomic numbers")
    if int(np.min(values)) < 0 or int(np.max(values)) > MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"{context} must contain atomic numbers from 0 to {MAX_ATOMIC_NUMBER}"
        )
    return np.ascontiguousarray(values, dtype=np.uint16)


def require_finite_coordinates(values: np.ndarray, *, context: str) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{context} contains NaN or infinity")


def validate_text_atom_count(
    atom_count: int,
    *,
    available_bytes: int,
    minimum_atom_bytes: int,
    format_name: str,
) -> None:
    count = int(atom_count)
    byte_count = max(0, int(available_bytes))
    minimum = max(1, int(minimum_atom_bytes))
    if count <= 0:
        raise ValueError(f"{format_name} atom count must be positive")
    if count > byte_count // minimum:
        raise ValueError(
            f"{format_name} atom count {count} cannot fit in the remaining "
            f"{byte_count} file bytes"
        )


def cell_from_lengths_angles(
    a: float,
    b: float,
    c: float,
    alpha_degrees: float = 90.0,
    beta_degrees: float = 90.0,
    gamma_degrees: float = 90.0,
) -> np.ndarray:
    parameters = (a, b, c, alpha_degrees, beta_degrees, gamma_degrees)
    if not all(math.isfinite(float(value)) for value in parameters):
        raise ValueError("Cell parameters must be finite")
    if min(float(a), float(b), float(c)) <= 0.0:
        raise ValueError("Cell lengths must be positive")
    alpha, beta, gamma = np.radians(
        np.asarray([alpha_degrees, beta_degrees, gamma_degrees], dtype=np.float64)
    )
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1.0e-8:
        raise ValueError("Cell gamma angle produces a singular basis")
    vector_a = np.array([a, 0.0, 0.0], dtype=np.float64)
    vector_b = np.array([b * np.cos(gamma), b * sin_gamma, 0.0], dtype=np.float64)
    c_x = c * np.cos(beta)
    c_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    c_z_squared = max(0.0, c * c - c_x * c_x - c_y * c_y)
    vector_c = np.array([c_x, c_y, np.sqrt(c_z_squared)], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        cell = np.ascontiguousarray([vector_a, vector_b, vector_c], dtype=np.float32)
    require_finite_coordinates(cell, context="Cell basis")
    return cell
