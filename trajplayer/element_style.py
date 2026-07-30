from __future__ import annotations

import numpy as np


DEFAULT_COVALENT_RADIUS_ANGSTROM = 0.55
DEFAULT_VDW_RADIUS_ANGSTROM = 1.50


def atom_render_arrays(atom_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical radii and colors as static contiguous GPU buffers."""
    from ase.data import covalent_radii, vdw_radii
    from ase.data.colors import cpk_colors

    numbers = np.asarray(atom_numbers, dtype=np.int32)
    covalent = _radius_array(
        numbers,
        covalent_radii,
        fallback=DEFAULT_COVALENT_RADIUS_ANGSTROM,
    )
    vdw = _radius_array(numbers, vdw_radii, fallback=DEFAULT_VDW_RADIUS_ANGSTROM)

    color_table = np.asarray(cpk_colors, dtype=np.float32)
    colors = np.empty((numbers.shape[0], 3), dtype=np.float32)
    colors[:] = np.array([0.65, 0.72, 0.78], dtype=np.float32)
    valid_colors = (numbers > 0) & (numbers < len(color_table))
    colors[valid_colors] = color_table[numbers[valid_colors], :3]
    np.clip(colors, 0.0, 1.0, out=colors)

    return (
        np.ascontiguousarray(covalent, dtype=np.float32),
        np.ascontiguousarray(vdw, dtype=np.float32),
        np.ascontiguousarray(colors, dtype=np.float32),
    )


def atom_style_arrays(atom_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covalent, _vdw, colors = atom_render_arrays(atom_numbers)
    return covalent, colors


def _radius_array(atom_numbers: np.ndarray, table: np.ndarray, *, fallback: float) -> np.ndarray:
    numbers = np.asarray(atom_numbers, dtype=np.int32)
    radii = np.empty(numbers.shape, dtype=np.float32)
    radius_table = np.asarray(table, dtype=np.float32)
    valid = (numbers > 0) & (numbers < len(radius_table))
    radii[:] = np.float32(fallback)
    radii[valid] = radius_table[numbers[valid]]
    invalid = ~np.isfinite(radii) | (radii <= 0.0)
    radii[invalid] = np.float32(fallback)
    return radii
