from __future__ import annotations

from collections.abc import Callable

import numpy as np


BOND_DISTANCE_SCALE = 1.25
MIN_BOND_DISTANCE = 0.20

_COVALENT_RADII: np.ndarray | None = None
_DEFAULT_RADIUS = np.float32(0.77)
_DEFAULT_VALENCE = np.uint8(4)
_VALENCE_BY_ATOMIC_NUMBER = {
    1: 1,
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 1,
    18: 0,
    19: 1,
    20: 2,
    35: 1,
    53: 1,
}


def valence_caps_for(atom_numbers: np.ndarray) -> np.ndarray:
    numbers = np.asarray(atom_numbers, dtype=np.int32)
    caps = np.full(numbers.shape, _DEFAULT_VALENCE, dtype=np.uint8)
    caps[numbers <= 0] = 0
    for atomic_number, cap in _VALENCE_BY_ATOMIC_NUMBER.items():
        caps[numbers == atomic_number] = np.uint8(cap)
    return np.ascontiguousarray(caps, dtype=np.uint8)


def connected_components(atom_count: int, bonds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atom_count = int(atom_count)
    if atom_count < 0:
        raise ValueError("atom_count must be non-negative")
    pairs = np.asarray(bonds, dtype=np.int32)
    if pairs.size == 0:
        component_ids = np.arange(atom_count, dtype=np.int32)
        component_sizes = np.ones(atom_count, dtype=np.int32)
        return component_ids, component_sizes
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("bonds must have shape (M, 2)")
    if int(pairs.min()) < 0 or int(pairs.max()) >= atom_count:
        raise ValueError("bonds contains atom indices outside atom_count")

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as scipy_connected_components

    left = pairs[:, 0]
    right = pairs[:, 1]
    rows = np.concatenate((left, right))
    columns = np.concatenate((right, left))
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, columns)),
        shape=(atom_count, atom_count),
    ).tocsr()
    _component_count, component_ids = scipy_connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    component_ids = np.asarray(component_ids, dtype=np.int32)
    component_sizes = np.bincount(component_ids, minlength=int(component_ids.max()) + 1)
    return (
        np.ascontiguousarray(component_ids, dtype=np.int32),
        np.ascontiguousarray(component_sizes, dtype=np.int32),
    )


def infer_bonds(
    positions: np.ndarray,
    atom_numbers: np.ndarray,
    *,
    distance_scale: float = BOND_DISTANCE_SCALE,
    min_distance: float = MIN_BOND_DISTANCE,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    frame = np.ascontiguousarray(positions, dtype=np.float32)
    numbers = np.asarray(atom_numbers, dtype=np.int32)
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if numbers.shape != (frame.shape[0],):
        raise ValueError("atom_numbers must have shape (N,)")
    atom_count = int(frame.shape[0])
    if atom_count < 2:
        return np.empty((0, 2), dtype=np.int32)

    radii = _radii_for(numbers)
    caps = valence_caps_for(numbers)
    if not np.any(caps):
        return np.empty((0, 2), dtype=np.int32)

    candidate_distance2, candidate_left, candidate_right = _candidate_pairs(
        frame,
        radii,
        caps,
        distance_scale=float(distance_scale),
        min_distance=float(min_distance),
        cancelled=cancelled,
    )
    if candidate_distance2.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    degrees = np.zeros(atom_count, dtype=np.uint8)
    max_bond_count = min(
        int(candidate_distance2.shape[0]),
        int(np.sum(caps, dtype=np.int64) // 2),
    )
    bonds = np.empty((max_bond_count, 2), dtype=np.int32)
    bond_count = 0
    order = np.lexsort((candidate_right, candidate_left, candidate_distance2))
    for candidate_index in order:
        if cancelled is not None and cancelled():
            return np.empty((0, 2), dtype=np.int32)
        i = int(candidate_left[candidate_index])
        j = int(candidate_right[candidate_index])
        if degrees[i] >= caps[i] or degrees[j] >= caps[j]:
            continue
        degrees[i] += 1
        degrees[j] += 1
        bonds[bond_count] = (i, j)
        bond_count += 1
        if bond_count == max_bond_count:
            break

    if bond_count == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.ascontiguousarray(bonds[:bond_count], dtype=np.int32)


def bond_segments_for_frame(
    positions: np.ndarray,
    bonds: np.ndarray,
    atom_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = np.ascontiguousarray(positions, dtype=np.float32)
    pairs = np.asarray(bonds, dtype=np.int32)
    colors = np.ascontiguousarray(atom_colors, dtype=np.float32)
    if pairs.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("bonds must have shape (M, 2)")
    left = pairs[:, 0]
    right = pairs[:, 1]
    left_pos = frame[left]
    right_pos = frame[right]
    mid = (left_pos + right_pos) * np.float32(0.5)

    segment_count = int(pairs.shape[0]) * 2
    starts = np.empty((segment_count, 3), dtype=np.float32)
    ends = np.empty((segment_count, 3), dtype=np.float32)
    segment_colors = np.empty((segment_count, 3), dtype=np.float32)
    starts[0::2] = left_pos
    ends[0::2] = mid
    segment_colors[0::2] = colors[left]
    starts[1::2] = right_pos
    ends[1::2] = mid
    segment_colors[1::2] = colors[right]
    return (
        np.ascontiguousarray(starts, dtype=np.float32),
        np.ascontiguousarray(ends, dtype=np.float32),
        np.ascontiguousarray(segment_colors, dtype=np.float32),
    )


def _radii_for(numbers: np.ndarray) -> np.ndarray:
    global _COVALENT_RADII
    if _COVALENT_RADII is None:
        from ase.data import covalent_radii

        _COVALENT_RADII = np.asarray(covalent_radii, dtype=np.float32)
    radius_table = _COVALENT_RADII
    radii = np.full(numbers.shape, _DEFAULT_RADIUS, dtype=np.float32)
    valid = (numbers > 0) & (numbers < len(radius_table))
    radii[valid] = radius_table[numbers[valid]]
    radii[~np.isfinite(radii)] = _DEFAULT_RADIUS
    radii[radii <= 0.0] = _DEFAULT_RADIUS
    return np.ascontiguousarray(radii, dtype=np.float32)


def _candidate_pairs(
    positions: np.ndarray,
    radii: np.ndarray,
    caps: np.ndarray,
    *,
    distance_scale: float,
    min_distance: float,
    cancelled: Callable[[], bool] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    active_indices = np.flatnonzero(caps > 0).astype(np.int32, copy=False)
    if active_indices.size < 2:
        return _empty_candidates()
    if cancelled is not None and cancelled():
        return _empty_candidates()

    maximum_cutoff = max(
        float(np.max(radii[active_indices]) * 2.0 * distance_scale),
        min_distance,
    )
    tree = cKDTree(positions[active_indices], compact_nodes=True, balanced_tree=True)
    local_pairs = tree.query_pairs(maximum_cutoff, output_type="ndarray")
    if local_pairs.size == 0:
        return _empty_candidates()
    if cancelled is not None and cancelled():
        return _empty_candidates()

    left = np.asarray(active_indices[local_pairs[:, 0]], dtype=np.int32)
    right = np.asarray(active_indices[local_pairs[:, 1]], dtype=np.int32)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    delta = positions[high] - positions[low]
    distance2 = np.einsum("ij,ij->i", delta, delta, dtype=np.float32)
    min_distance2 = float(min_distance * min_distance)
    cutoff = (radii[low] + radii[high]) * np.float32(distance_scale)
    accepted = (distance2 >= np.float32(min_distance2)) & (distance2 <= cutoff * cutoff)
    return (
        np.ascontiguousarray(distance2[accepted], dtype=np.float32),
        np.ascontiguousarray(low[accepted], dtype=np.int32),
        np.ascontiguousarray(high[accepted], dtype=np.int32),
    )


def _empty_candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
    )
