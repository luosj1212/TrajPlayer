from __future__ import annotations

import itertools

import numpy as np


try:
    from . import _trajcore as _native
except ImportError as exc:
    _native = None
    NATIVE_IMPORT_ERROR: ImportError | None = exc
else:
    NATIVE_IMPORT_ERROR = None


NATIVE_AVAILABLE = _native is not None
NATIVE_DEPTH_ORDER_AVAILABLE = _native is not None and hasattr(_native, "coarse_depth_order")
NATIVE_POSITION_DEPTH_ORDER_AVAILABLE = _native is not None and hasattr(
    _native,
    "coarse_position_depth_order",
)
NATIVE_XYZ_READ_AVAILABLE = _native is not None and hasattr(_native, "xyz_read_frame_into")
NATIVE_VALENCE_SELECTION_AVAILABLE = _native is not None and hasattr(
    _native,
    "select_valence_bonds",
)
NATIVE_CONNECTED_COMPONENTS_AVAILABLE = _native is not None and hasattr(
    _native,
    "connected_components",
)
NATIVE_CANDIDATE_PAIRS_AVAILABLE = _native is not None and hasattr(
    _native,
    "candidate_pairs",
)
NATIVE_FULL_AVAILABLE = all(
    (
        NATIVE_AVAILABLE,
        NATIVE_DEPTH_ORDER_AVAILABLE,
        NATIVE_POSITION_DEPTH_ORDER_AVAILABLE,
        NATIVE_XYZ_READ_AVAILABLE,
        NATIVE_VALENCE_SELECTION_AVAILABLE,
        NATIVE_CONNECTED_COMPONENTS_AVAILABLE,
        NATIVE_CANDIDATE_PAIRS_AVAILABLE,
    )
)
DEPTH_BIN_COUNT = 256


def xyz_read_frame_into(
    source,
    *,
    data_offset: int,
    data_end: int,
    positions: np.ndarray,
    identity_column: int,
    identity_is_atomic_number: bool,
    position_columns: tuple[int, int, int],
    expected_columns: int,
    expected_atom_numbers: np.ndarray,
) -> bool:
    """Parse XYZ atom rows directly into a caller-owned contiguous frame buffer."""

    layout = np.asarray(
        [
            int(identity_column),
            int(position_columns[0]),
            int(position_columns[1]),
            int(position_columns[2]),
            int(expected_columns),
        ],
        dtype=np.int32,
    )
    expected_values = np.asarray(expected_atom_numbers)
    if (
        expected_values.ndim != 1
        or expected_values.dtype.kind not in {"i", "u"}
        or (
            expected_values.size
            and (
                int(np.min(expected_values)) < 0
                or int(np.max(expected_values)) > 118
            )
        )
    ):
        raise ValueError("expected_atom_numbers must contain integers from 0 to 118")
    expected = np.ascontiguousarray(expected_values, dtype=np.uint16)
    if not NATIVE_XYZ_READ_AVAILABLE:
        return False
    _native.xyz_read_frame_into(
        source,
        int(data_offset),
        int(data_end),
        positions,
        layout,
        expected,
        bool(identity_is_atomic_number),
    )
    return True


def coarse_depth_order(view_depth: np.ndarray) -> np.ndarray:
    """Return the existing 256-bin far-to-near order, using native O(N) counting."""

    depth = np.asarray(view_depth, dtype=np.float32)
    if depth.ndim != 1:
        raise ValueError("view_depth must be one-dimensional")
    if _native is not None and hasattr(_native, "coarse_depth_order"):
        return np.ascontiguousarray(_native.coarse_depth_order(depth), dtype=np.int64)
    return _python_coarse_depth_order(depth)


def coarse_position_depth_order(
    positions: np.ndarray,
    camera_forward: np.ndarray,
) -> np.ndarray:
    """Project canonical positions and return the visual coarse depth order."""

    frame = np.ascontiguousarray(positions, dtype=np.float32)
    forward = np.ascontiguousarray(camera_forward, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if forward.shape != (3,):
        raise ValueError("camera_forward must have shape (3,)")
    if NATIVE_POSITION_DEPTH_ORDER_AVAILABLE:
        return np.ascontiguousarray(
            _native.coarse_position_depth_order(frame, forward),
            dtype=np.int64,
        )
    return coarse_depth_order(frame @ forward)


def _python_coarse_depth_order(depth: np.ndarray) -> np.ndarray:
    if depth.size <= 1:
        return np.arange(depth.size, dtype=np.int64)
    minimum = float(np.min(depth))
    maximum = float(np.max(depth))
    span = maximum - minimum
    if not np.isfinite(span) or span <= np.finfo(np.float32).eps:
        return np.arange(depth.size - 1, -1, -1, dtype=np.int64)
    scaled = (depth - minimum) * ((DEPTH_BIN_COUNT - 1) / span)
    bins = np.asarray(np.clip(scaled, 0, DEPTH_BIN_COUNT - 1), dtype=np.uint8)
    return np.argsort(bins, kind="stable")[::-1]


def connected_components(
    atom_count: int,
    bonds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.ascontiguousarray(bonds, dtype=np.int32)
    if _native is not None:
        labels, sizes = _native.connected_components(int(atom_count), pairs)
        return (
            np.ascontiguousarray(labels, dtype=np.int32),
            np.ascontiguousarray(sizes, dtype=np.int32),
        )
    return _python_connected_components(int(atom_count), pairs)


def candidate_pairs(
    positions: np.ndarray,
    active_indices: np.ndarray,
    maximum_cutoff: float,
    *,
    cell: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = np.ascontiguousarray(positions, dtype=np.float32)
    active = np.ascontiguousarray(active_indices, dtype=np.int32)
    matrix = None if cell is None else np.ascontiguousarray(cell, dtype=np.float64)
    if _native is not None:
        distance2, left, right = _native.candidate_pairs(
            frame,
            active,
            float(maximum_cutoff),
            matrix,
        )
        return (
            np.ascontiguousarray(distance2, dtype=np.float32),
            np.ascontiguousarray(left, dtype=np.int32),
            np.ascontiguousarray(right, dtype=np.int32),
        )
    return _python_candidate_pairs(
        frame,
        active,
        float(maximum_cutoff),
        cell=matrix,
    )


def select_valence_bonds(
    distance2: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    """Select shortest candidate bonds while respecting per-atom valence caps."""

    distances = np.ascontiguousarray(distance2, dtype=np.float32)
    low = np.ascontiguousarray(left, dtype=np.int32)
    high = np.ascontiguousarray(right, dtype=np.int32)
    limits = np.ascontiguousarray(caps, dtype=np.uint8)
    _validate_valence_selection_inputs(distances, low, high, limits)
    if NATIVE_VALENCE_SELECTION_AVAILABLE:
        return np.ascontiguousarray(
            _native.select_valence_bonds(distances, low, high, limits),
            dtype=np.int32,
        )
    return _python_select_valence_bonds(distances, low, high, limits)


def _validate_valence_selection_inputs(
    distance2: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    caps: np.ndarray,
) -> None:
    if distance2.ndim != 1 or left.ndim != 1 or right.ndim != 1 or caps.ndim != 1:
        raise ValueError("distance2, left, right, and caps must be one-dimensional")
    if distance2.shape != left.shape or distance2.shape != right.shape:
        raise ValueError("candidate arrays must have matching lengths")
    if not np.all(np.isfinite(distance2)):
        raise ValueError("candidate distances must be finite")
    if left.size and (
        int(np.min(left)) < 0
        or int(np.max(left)) >= caps.size
        or int(np.min(right)) < 0
        or int(np.max(right)) >= caps.size
    ):
        raise ValueError("candidate atom indices must be in range")
    if np.any(left == right):
        raise ValueError("candidate atom indices must be distinct")


def _python_select_valence_bonds(
    distance2: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    if distance2.size == 0 or not np.any(caps):
        return np.empty((0, 2), dtype=np.int32)
    degrees = np.zeros(caps.shape, dtype=np.uint16)
    maximum = min(int(distance2.size), int(np.sum(caps, dtype=np.int64) // 2))
    bonds = np.empty((maximum, 2), dtype=np.int32)
    count = 0
    order = np.lexsort((right, left, distance2))
    for candidate in order:
        i = int(left[candidate])
        j = int(right[candidate])
        if degrees[i] >= caps[i] or degrees[j] >= caps[j]:
            continue
        degrees[i] += 1
        degrees[j] += 1
        bonds[count] = (i, j)
        count += 1
        if count == maximum:
            break
    return np.ascontiguousarray(bonds[:count], dtype=np.int32)


def _python_connected_components(
    atom_count: int,
    bonds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if atom_count < 0:
        raise ValueError("atom_count must be non-negative")
    if bonds.size == 0:
        return np.arange(atom_count, dtype=np.int32), np.ones(atom_count, dtype=np.int32)
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError("bonds must have shape (M, 2)")
    if int(bonds.min()) < 0 or int(bonds.max()) >= atom_count:
        raise ValueError("bonds contains atom indices outside atom_count")

    parent = np.full(atom_count, -1, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] >= 0:
            root = int(parent[root])
        while node != root:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    for left_value, right_value in bonds:
        left = find(int(left_value))
        right = find(int(right_value))
        if left == right:
            continue
        if parent[left] > parent[right]:
            left, right = right, left
        parent[left] += parent[right]
        parent[right] = left

    labels = np.empty(atom_count, dtype=np.int32)
    root_labels: dict[int, int] = {}
    for atom in range(atom_count):
        root = find(atom)
        labels[atom] = root_labels.setdefault(root, len(root_labels))
    sizes = np.bincount(labels, minlength=len(root_labels)).astype(np.int32, copy=False)
    return labels, np.ascontiguousarray(sizes, dtype=np.int32)


def _python_candidate_pairs(
    positions: np.ndarray,
    active_indices: np.ndarray,
    maximum_cutoff: float,
    *,
    cell: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(maximum_cutoff) or maximum_cutoff <= 0.0:
        raise ValueError("maximum_cutoff must be positive and finite")
    if active_indices.size < 2:
        return _empty_candidates()
    if active_indices.ndim != 1:
        raise ValueError("active_indices must be one-dimensional")
    if int(active_indices.min()) < 0 or int(active_indices.max()) >= positions.shape[0]:
        raise ValueError("active_indices contains an out-of-range atom")
    if not np.all(np.isfinite(positions[active_indices])):
        raise ValueError("active atom positions must be finite")
    if cell is not None:
        return _python_periodic_candidate_pairs(
            positions,
            active_indices,
            cell,
            maximum_cutoff,
        )

    origin = np.min(positions[active_indices], axis=0)
    coordinates = np.floor(
        (positions[active_indices] - origin) / np.float32(maximum_cutoff)
    ).astype(np.int64)
    cells: dict[tuple[int, int, int], np.ndarray] = {}
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    sorted_coordinates = coordinates[order]
    boundaries = np.flatnonzero(
        np.any(sorted_coordinates[1:] != sorted_coordinates[:-1], axis=1)
    ) + 1
    for group in np.split(order, boundaries):
        coordinate = tuple(int(value) for value in coordinates[group[0]])
        cells[coordinate] = active_indices[group]

    cutoff2 = np.float32(maximum_cutoff * maximum_cutoff)
    distances: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    half_neighbors = tuple(
        offset
        for offset in itertools.product((-1, 0, 1), repeat=3)
        if offset >= (0, 0, 0)
    )
    for coordinate, left_atoms in cells.items():
        for offset in half_neighbors:
            neighbor_coordinate = tuple(coordinate[axis] + offset[axis] for axis in range(3))
            right_atoms = cells.get(neighbor_coordinate)
            if right_atoms is None:
                continue
            if offset == (0, 0, 0):
                local_left, local_right = np.triu_indices(left_atoms.size, k=1)
                left = left_atoms[local_left]
                right = left_atoms[local_right]
            else:
                left = np.repeat(left_atoms, right_atoms.size)
                right = np.tile(right_atoms, left_atoms.size)
            if left.size == 0:
                continue
            delta = positions[right] - positions[left]
            distance2 = np.einsum("ij,ij->i", delta, delta, dtype=np.float32)
            keep = distance2 <= cutoff2
            if np.any(keep):
                distances.append(distance2[keep])
                left_parts.append(np.minimum(left[keep], right[keep]))
                right_parts.append(np.maximum(left[keep], right[keep]))
    if not distances:
        return _empty_candidates()
    return (
        np.ascontiguousarray(np.concatenate(distances), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(left_parts), dtype=np.int32),
        np.ascontiguousarray(np.concatenate(right_parts), dtype=np.int32),
    )


def _python_periodic_candidate_pairs(
    positions: np.ndarray,
    active_indices: np.ndarray,
    cell: np.ndarray,
    maximum_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("cell must be a finite (3, 3) matrix")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cell must be invertible") from exc
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    if np.any(reciprocal_norms <= 0.0) or not np.all(np.isfinite(reciprocal_norms)):
        raise ValueError("cell must be invertible")
    heights = 1.0 / reciprocal_norms
    bin_counts = np.maximum(1, np.floor(heights / maximum_cutoff).astype(np.int64))

    active_positions = np.asarray(positions[active_indices], dtype=np.float64)
    fractional = active_positions @ inverse
    fractional -= np.floor(fractional)
    coordinates = np.floor(fractional * bin_counts).astype(np.int64)
    np.minimum(coordinates, bin_counts - 1, out=coordinates)

    cells: dict[tuple[int, int, int], np.ndarray] = {}
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    sorted_coordinates = coordinates[order]
    boundaries = np.flatnonzero(
        np.any(sorted_coordinates[1:] != sorted_coordinates[:-1], axis=1)
    ) + 1
    for group in np.split(order, boundaries):
        coordinate = tuple(int(value) for value in coordinates[group[0]])
        cells[coordinate] = active_indices[group]

    cutoff2 = np.float32(maximum_cutoff * maximum_cutoff)
    distances: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for coordinate, left_atoms in cells.items():
        visited_neighbors: set[tuple[int, int, int]] = set()
        for offset in offsets:
            neighbor_coordinate = tuple(
                (coordinate[axis] + offset[axis]) % int(bin_counts[axis])
                for axis in range(3)
            )
            if neighbor_coordinate in visited_neighbors or neighbor_coordinate < coordinate:
                continue
            visited_neighbors.add(neighbor_coordinate)
            right_atoms = cells.get(neighbor_coordinate)
            if right_atoms is None:
                continue
            if neighbor_coordinate == coordinate:
                local_left, local_right = np.triu_indices(left_atoms.size, k=1)
                left = left_atoms[local_left]
                right = left_atoms[local_right]
            else:
                left = np.repeat(left_atoms, right_atoms.size)
                right = np.tile(right_atoms, left_atoms.size)
            if left.size == 0:
                continue
            delta = np.asarray(positions[right] - positions[left], dtype=np.float64)
            delta_fractional = delta @ inverse
            delta_fractional -= np.rint(delta_fractional)
            delta = delta_fractional @ matrix
            distance2 = np.asarray(np.einsum("ij,ij->i", delta, delta), dtype=np.float32)
            keep = distance2 <= cutoff2
            if np.any(keep):
                distances.append(distance2[keep])
                left_parts.append(np.minimum(left[keep], right[keep]))
                right_parts.append(np.maximum(left[keep], right[keep]))
    if not distances:
        return _empty_candidates()
    return (
        np.ascontiguousarray(np.concatenate(distances), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(left_parts), dtype=np.int32),
        np.ascontiguousarray(np.concatenate(right_parts), dtype=np.int32),
    )


def _empty_candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
    )
