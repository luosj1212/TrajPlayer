from __future__ import annotations

import math

import numpy as np

from .pbc import minimum_image_displacement


def _vector(left: np.ndarray, right: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    displacement = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if displacement.shape != (3,):
        raise ValueError("Coordinates must each have shape (3,)")
    if cell is not None:
        displacement = minimum_image_displacement(displacement, cell)
    return displacement


def distance(left: np.ndarray, right: np.ndarray, cell: np.ndarray | None = None) -> float:
    return float(np.linalg.norm(_vector(left, right, cell)))


def angle(
    first: np.ndarray,
    vertex: np.ndarray,
    third: np.ndarray,
    cell: np.ndarray | None = None,
) -> float:
    left = _vector(first, vertex, cell)
    right = _vector(third, vertex, cell)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-15:
        raise ValueError("Cannot measure an angle with a zero-length arm")
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def dihedral(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    cell: np.ndarray | None = None,
) -> float:
    b0 = _vector(first, second, cell)
    b1 = _vector(third, second, cell)
    b2 = _vector(fourth, third, cell)
    norm_b1 = float(np.linalg.norm(b1))
    if norm_b1 <= 1.0e-15:
        raise ValueError("Cannot measure a dihedral with a zero-length central bond")
    b1n = b1 / norm_b1
    projected_first = b0 - np.dot(b0, b1n) * b1n
    projected_last = b2 - np.dot(b2, b1n) * b1n
    if np.linalg.norm(projected_first) <= 1.0e-15 or np.linalg.norm(projected_last) <= 1.0e-15:
        raise ValueError("Cannot measure a dihedral for collinear bond vectors")
    x = float(np.dot(projected_first, projected_last))
    y = float(np.dot(np.cross(b1n, projected_first), projected_last))
    return math.degrees(math.atan2(y, x))


def _validated_weights(count: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (count,):
        raise ValueError("weights must contain one value per position")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if float(values.sum()) <= 0.0:
        raise ValueError("weights must have a positive total")
    return values


def center_of_mass(positions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("positions must have shape (N, 3) with N > 0")
    mass = _validated_weights(values.shape[0], weights)
    return np.ascontiguousarray(
        np.sum(values * mass[:, None], axis=0) / float(mass.sum()),
        dtype=np.float64,
    )


def radius_of_gyration(positions: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(positions, dtype=np.float64)
    mass = _validated_weights(values.shape[0], weights)
    center = center_of_mass(values, mass)
    squared = np.einsum("ij,ij->i", values - center, values - center)
    return math.sqrt(float(np.dot(mass, squared) / mass.sum()))
