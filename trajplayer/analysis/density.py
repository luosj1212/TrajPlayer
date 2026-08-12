from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .pbc import cartesian_to_fractional, cell_volume, wrap_fractional


AMU_PER_ANGSTROM3_TO_G_CM3 = 1.66053906660


def bulk_density(
    cells: Iterable[np.ndarray],
    total_weight: float,
    *,
    mass_density: bool,
) -> np.ndarray:
    weight = float(total_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("total_weight must be finite and non-negative")
    values = np.asarray([weight / cell_volume(cell) for cell in cells], dtype=np.float64)
    if mass_density:
        values *= AMU_PER_ANGSTROM3_TO_G_CM3
    return np.ascontiguousarray(values, dtype=np.float64)


def density_profile(
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    axis: int,
    bins: int,
    weights: np.ndarray | None = None,
    mass_density: bool = False,
) -> np.ndarray:
    frame = np.asarray(positions, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    selected_axis = int(axis)
    if selected_axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2")
    bin_count = int(bins)
    if bin_count <= 0:
        raise ValueError("bins must be positive")
    histogram_weights = None
    if weights is not None:
        histogram_weights = np.asarray(weights, dtype=np.float64)
        if histogram_weights.shape != (frame.shape[0],):
            raise ValueError("weights must contain one value per position")
        if not np.all(np.isfinite(histogram_weights)):
            raise ValueError("weights contain a non-finite value")
    fractional = wrap_fractional(cartesian_to_fractional(frame, cell))
    histogram, _edges = np.histogram(
        fractional[:, selected_axis],
        bins=bin_count,
        range=(0.0, 1.0),
        weights=histogram_weights,
    )
    profile = histogram.astype(np.float64, copy=False) / (cell_volume(cell) / bin_count)
    if mass_density:
        profile *= AMU_PER_ANGSTROM3_TO_G_CM3
    return np.ascontiguousarray(profile, dtype=np.float64)
