from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .alignment import align_positions, rmsd


def rmsd_series(
    frames: Iterable[np.ndarray],
    reference: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    translate: bool = True,
    rotate: bool = True,
) -> np.ndarray:
    values = [
        rmsd(
            frame,
            reference,
            weights=weights,
            translate=translate,
            rotate=rotate,
        )
        for frame in frames
    ]
    return np.ascontiguousarray(values, dtype=np.float64)


def rmsf(
    frames: Iterable[np.ndarray],
    *,
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    translate: bool = True,
    rotate: bool = True,
) -> np.ndarray:
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    count = 0
    target = None if reference is None else np.asarray(reference, dtype=np.float64)
    for frame in frames:
        values = np.asarray(frame, dtype=np.float64)
        if target is not None:
            values = align_positions(
                values,
                target,
                weights=weights,
                translate=translate,
                rotate=rotate,
            )
        if mean is None:
            mean = np.zeros_like(values, dtype=np.float64)
            m2 = np.zeros_like(values, dtype=np.float64)
        if values.shape != mean.shape:
            raise ValueError("All RMSF frames must have the same shape")
        count += 1
        delta = values - mean
        mean += delta / count
        delta2 = values - mean
        m2 += delta * delta2
    if count == 0 or mean is None or m2 is None:
        raise ValueError("RMSF requires at least one frame")
    return np.sqrt(np.sum(m2 / count, axis=1, dtype=np.float64))
