from __future__ import annotations

import numpy as np


def validated_cell(cell: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cell contains a non-finite value")
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1.0e-12:
        raise ValueError("cell is singular")
    return matrix


def cell_volume(cell: np.ndarray) -> float:
    return abs(float(np.linalg.det(validated_cell(cell))))


def cartesian_to_fractional(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("positions must end with three Cartesian coordinates")
    return np.ascontiguousarray(values @ np.linalg.inv(validated_cell(cell)), dtype=np.float64)


def fractional_to_cartesian(fractional: np.ndarray, cell: np.ndarray) -> np.ndarray:
    values = np.asarray(fractional, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("fractional coordinates must end with three values")
    return np.ascontiguousarray(values @ validated_cell(cell), dtype=np.float64)


def minimum_image_displacement(displacement: np.ndarray, cell: np.ndarray) -> np.ndarray:
    values = np.asarray(displacement, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("displacement must end with three Cartesian coordinates")
    matrix = validated_cell(cell)
    fractional = values @ np.linalg.inv(matrix)
    fractional -= np.rint(fractional)
    return np.ascontiguousarray(fractional @ matrix, dtype=np.float64)


def wrap_fractional(fractional: np.ndarray) -> np.ndarray:
    values = np.asarray(fractional, dtype=np.float64)
    return np.ascontiguousarray(values - np.floor(values), dtype=np.float64)


def unwrap_fractional_step(
    previous_fractional: np.ndarray,
    current_fractional: np.ndarray,
    previous_unwrapped: np.ndarray,
    current_cell: np.ndarray,
) -> np.ndarray:
    previous = np.asarray(previous_fractional, dtype=np.float64)
    current = np.asarray(current_fractional, dtype=np.float64)
    unwrapped = np.asarray(previous_unwrapped, dtype=np.float64)
    if previous.shape != current.shape or current.shape != unwrapped.shape:
        raise ValueError("unwrap inputs must have matching shapes")
    fractional_delta = current - previous
    fractional_delta -= np.rint(fractional_delta)
    return np.ascontiguousarray(
        unwrapped + fractional_delta @ validated_cell(current_cell),
        dtype=np.float64,
    )


def make_whole_relative_to_anchor(
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    anchor: int = 0,
) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if values.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    anchor_index = int(anchor)
    if anchor_index < 0 or anchor_index >= values.shape[0]:
        raise IndexError(anchor)
    origin = values[anchor_index]
    whole = origin + minimum_image_displacement(values - origin, cell)
    return np.ascontiguousarray(whole, dtype=np.float64)
