from __future__ import annotations

import math

import numpy as np


def _weights(count: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (count,):
        raise ValueError("weights must contain one value per atom")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have a positive total")
    return values


def kabsch_rotation(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    current = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if current.shape != target.shape or current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("mobile and reference must both have shape (N, 3)")
    mass = _weights(current.shape[0], weights)
    covariance = (current * mass[:, None]).T @ target
    left, _singular, right_transpose = np.linalg.svd(covariance)
    if float(np.linalg.det(left @ right_transpose)) < 0.0:
        left[:, -1] *= -1.0
    return np.ascontiguousarray(left @ right_transpose, dtype=np.float64)


def align_positions(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    translate: bool = True,
    rotate: bool = True,
) -> np.ndarray:
    current = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if current.shape != target.shape or current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("mobile and reference must both have shape (N, 3)")
    mass = _weights(current.shape[0], weights)
    current_center = np.average(current, axis=0, weights=mass) if translate else np.zeros(3)
    target_center = np.average(target, axis=0, weights=mass) if translate else np.zeros(3)
    centered = current - current_center
    target_centered = target - target_center
    if rotate:
        centered = centered @ kabsch_rotation(centered, target_centered, mass)
    return np.ascontiguousarray(centered + target_center, dtype=np.float64)


def rmsd(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    translate: bool = True,
    rotate: bool = True,
) -> float:
    target = np.asarray(reference, dtype=np.float64)
    aligned = align_positions(
        mobile,
        target,
        weights=weights,
        translate=translate,
        rotate=rotate,
    )
    mass = _weights(target.shape[0], weights)
    squared = np.einsum("ij,ij->i", aligned - target, aligned - target)
    return math.sqrt(float(np.dot(mass, squared) / mass.sum()))
