from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorSceneSnapshot:
    """Immutable CPU snapshot of the molecular viewport at export time."""

    width: int
    height: int
    view_matrix: np.ndarray
    projection_matrix: np.ndarray
    positions: np.ndarray
    atom_indices: np.ndarray
    atom_radii: np.ndarray
    atom_colors: np.ndarray
    atom_selected: np.ndarray
    selection_color: tuple[float, float, float]
    bond_pairs: np.ndarray
    bond_colors: np.ndarray
    atom_size_scale: float
    bond_size_scale: float
    bond_radius: float
    bond_endpoint_radius_scale: float
    show_atoms: bool
    show_bonds: bool
    periodic_cell: np.ndarray | None
    box_segments: np.ndarray
    box_color: tuple[float, float, float]
    background: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _ProjectedPoints:
    screen: np.ndarray
    depth: np.ndarray
    clip_w: np.ndarray
    ndc_x: np.ndarray
    valid: np.ndarray


def write_molecule_svg(
    path: Path,
    scene: VectorSceneSnapshot,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Write the current molecular viewport as editable SVG primitives."""

    _validate_scene(scene)
    if cancelled is not None and cancelled():
        raise RuntimeError("Export cancelled")

    width = int(scene.width)
    height = int(scene.height)
    atom_projection = _project_points(
        scene.positions,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    atom_rx, atom_ry = _projected_radii(
        scene.atom_radii * np.float32(scene.atom_size_scale),
        atom_projection,
        scene.projection_matrix,
        width,
        height,
    )
    atom_valid = atom_projection.valid.copy()
    atom_valid &= atom_rx > 0.0
    atom_valid &= atom_ry > 0.0
    atom_valid &= atom_projection.screen[:, 0] + atom_rx >= 0.0
    atom_valid &= atom_projection.screen[:, 0] - atom_rx <= width
    atom_valid &= atom_projection.screen[:, 1] + atom_ry >= 0.0
    atom_valid &= atom_projection.screen[:, 1] - atom_ry <= height

    bond_start, bond_end = _bond_endpoints(scene)
    bond_mid = (bond_start + bond_end) * np.float32(0.5)
    bond_start_projection = _project_points(
        bond_start,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    bond_end_projection = _project_points(
        bond_end,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    bond_mid_projection = _project_points(
        bond_mid,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    bond_rx, _bond_ry = _projected_radii(
        np.full(
            bond_mid.shape[0],
            np.float32(scene.bond_radius * scene.bond_size_scale),
            dtype=np.float32,
        ),
        bond_mid_projection,
        scene.projection_matrix,
        width,
        height,
    )
    bond_delta_screen = bond_end_projection.screen - bond_start_projection.screen
    bond_length_screen = np.linalg.norm(bond_delta_screen, axis=1)
    bond_valid = bond_start_projection.valid & bond_end_projection.valid
    bond_valid &= bond_mid_projection.valid
    bond_valid &= bond_rx > 0.0
    bond_valid &= bond_length_screen > 1.0e-4

    box_start = scene.box_segments[:, 0] if scene.box_segments.size else np.empty((0, 3))
    box_end = scene.box_segments[:, 1] if scene.box_segments.size else np.empty((0, 3))
    box_mid = (box_start + box_end) * 0.5
    box_start_projection = _project_points(
        box_start,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    box_end_projection = _project_points(
        box_end,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    box_mid_projection = _project_points(
        box_mid,
        scene.view_matrix,
        scene.projection_matrix,
        width,
        height,
    )
    box_valid = box_start_projection.valid & box_end_projection.valid
    box_valid &= box_mid_projection.clip_w > 1.0e-12

    atom_indices = np.flatnonzero(atom_valid) if scene.show_atoms else np.empty(0, dtype=np.int64)
    bond_indices = np.flatnonzero(bond_valid) if scene.show_bonds else np.empty(0, dtype=np.int64)
    box_indices = np.flatnonzero(box_valid)
    primitive_kind = np.concatenate(
        (
            np.zeros(box_indices.size, dtype=np.uint8),
            np.ones(bond_indices.size, dtype=np.uint8),
            np.full(atom_indices.size, 2, dtype=np.uint8),
        )
    )
    primitive_index = np.concatenate((box_indices, bond_indices, atom_indices))
    primitive_depth = np.concatenate(
        (
            box_mid_projection.depth[box_indices],
            bond_mid_projection.depth[bond_indices],
            atom_projection.depth[atom_indices],
        )
    )
    primitive_priority = primitive_kind
    draw_order = np.lexsort((primitive_priority, -primitive_depth))

    atom_rgb = _rgb8_array(scene.atom_colors)
    atom_selected = np.asarray(scene.atom_selected, dtype=np.bool_)
    bond_rgb = _rgb8_array(
        np.asarray(scene.bond_colors).reshape(-1, 3)
    ).reshape(-1, 2, 3)
    with Path(path).open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
        _write_header(handle, scene, atom_rgb, atom_selected, bond_rgb)
        handle.write('<g clip-path="url(#viewport-clip)">\n')
        for draw_number, primitive in enumerate(draw_order):
            if draw_number % 4096 == 0 and cancelled is not None and cancelled():
                raise RuntimeError("Export cancelled")
            kind = int(primitive_kind[primitive])
            index = int(primitive_index[primitive])
            if kind == 0:
                _write_box_edge(
                    handle,
                    box_start_projection.screen[index],
                    box_end_projection.screen[index],
                    scene.box_color,
                )
            elif kind == 1:
                _write_bond(
                    handle,
                    bond_start_projection.screen[index],
                    bond_end_projection.screen[index],
                    float(bond_rx[index]),
                    (
                        _bond_style_id(bond_rgb[index, 0]),
                        _bond_style_id(bond_rgb[index, 1]),
                    ),
                )
            else:
                _write_atom(
                    handle,
                    atom_projection.screen[index],
                    float(atom_rx[index]),
                    float(atom_ry[index]),
                    _atom_style_id(atom_rgb[index], bool(atom_selected[index])),
                    int(scene.atom_indices[index]),
                )
        handle.write("</g>\n</svg>\n")


def _validate_scene(scene: VectorSceneSnapshot) -> None:
    positions = np.asarray(scene.positions)
    atom_count = int(positions.shape[0]) if positions.ndim == 2 else -1
    if scene.width <= 0 or scene.height <= 0:
        raise ValueError("SVG viewport dimensions must be positive")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if np.asarray(scene.atom_indices).shape != (atom_count,):
        raise ValueError("atom_indices must have shape (N,)")
    if np.asarray(scene.atom_radii).shape != (atom_count,):
        raise ValueError("atom_radii must have shape (N,)")
    if np.asarray(scene.atom_colors).shape != (atom_count, 3):
        raise ValueError("atom_colors must have shape (N, 3)")
    if np.asarray(scene.atom_selected).shape != (atom_count,):
        raise ValueError("atom_selected must have shape (N,)")
    pairs = np.asarray(scene.bond_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("bond_pairs must have shape (M, 2)")
    if pairs.size and (int(pairs.min()) < 0 or int(pairs.max()) >= atom_count):
        raise ValueError("bond_pairs contains an atom outside the scene")
    if np.asarray(scene.bond_colors).shape != (pairs.shape[0], 2, 3):
        raise ValueError("bond_colors must have shape (M, 2, 3)")
    if np.asarray(scene.view_matrix).shape != (4, 4):
        raise ValueError("view_matrix must have shape (4, 4)")
    if np.asarray(scene.projection_matrix).shape != (4, 4):
        raise ValueError("projection_matrix must have shape (4, 4)")
    segments = np.asarray(scene.box_segments)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError("box_segments must have shape (K, 2, 3)")


def _project_points(
    points: np.ndarray,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    width: int,
    height: int,
) -> _ProjectedPoints:
    values = np.asarray(points, dtype=np.float64)
    count = int(values.shape[0])
    screen = np.empty((count, 2), dtype=np.float32)
    depth = np.empty(count, dtype=np.float32)
    clip_w = np.empty(count, dtype=np.float32)
    ndc_x = np.empty(count, dtype=np.float32)
    valid = np.zeros(count, dtype=np.bool_)
    view = np.asarray(view_matrix, dtype=np.float64)
    projection = np.asarray(projection_matrix, dtype=np.float64)
    for start in range(0, count, 131_072):
        stop = min(count, start + 131_072)
        homogeneous = np.ones((stop - start, 4), dtype=np.float64)
        homogeneous[:, :3] = values[start:stop]
        view_points = homogeneous @ view.T
        clip = view_points @ projection.T
        w = clip[:, 3]
        good_w = w > 1.0e-12
        safe_w = np.where(good_w, w, 1.0)
        ndc = clip[:, :3] / safe_w[:, None]
        screen[start:stop, 0] = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.float32)
        screen[start:stop, 1] = ((1.0 - ndc[:, 1]) * 0.5 * height).astype(np.float32)
        depth[start:stop] = ndc[:, 2].astype(np.float32)
        clip_w[start:stop] = w.astype(np.float32)
        ndc_x[start:stop] = ndc[:, 0].astype(np.float32)
        valid[start:stop] = good_w & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    return _ProjectedPoints(screen, depth, clip_w, ndc_x, valid)


def _projected_radii(
    radii: np.ndarray,
    projected: _ProjectedPoints,
    projection_matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(radii, dtype=np.float64)
    projection = np.asarray(projection_matrix, dtype=np.float64)
    edge_x = projected.ndc_x.astype(np.float64)
    edge_w = projected.clip_w.astype(np.float64) + projection[3, 0] * values
    valid_w = edge_w > 1.0e-12
    edge_x = np.where(
        valid_w,
        (
            projected.ndc_x.astype(np.float64)
            * projected.clip_w.astype(np.float64)
            + projection[0, 0] * values
        )
        / np.where(valid_w, edge_w, 1.0),
        projected.ndc_x,
    )
    radius_ndc_x = np.maximum(
        np.abs(edge_x - projected.ndc_x.astype(np.float64)),
        1.0e-6,
    )
    aspect = projection[1, 1] / max(projection[0, 0], 1.0e-4)
    rx = (radius_ndc_x * width * 0.5).astype(np.float32)
    ry = (radius_ndc_x * abs(aspect) * height * 0.5).astype(np.float32)
    rx[~projected.valid] = 0.0
    ry[~projected.valid] = 0.0
    return rx, ry


def _bond_endpoints(scene: VectorSceneSnapshot) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(scene.bond_pairs, dtype=np.int64)
    if pairs.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    positions = np.asarray(scene.positions, dtype=np.float32)
    source = positions[pairs[:, 0]]
    delta = positions[pairs[:, 1]] - source
    cell = scene.periodic_cell
    if cell is not None:
        matrix = np.asarray(cell, dtype=np.float64)
        if matrix.shape == (3, 3) and abs(float(np.linalg.det(matrix))) > 1.0e-10:
            fractional = delta.astype(np.float64) @ np.linalg.inv(matrix)
            fractional -= np.rint(fractional)
            delta = (fractional @ matrix).astype(np.float32)
    lengths = np.linalg.norm(delta, axis=1)
    safe_lengths = np.maximum(lengths, np.float32(1.0e-6))
    direction = delta / safe_lengths[:, None]
    limit = lengths * np.float32(0.45)
    endpoint_scale = np.float32(
        scene.atom_size_scale * scene.bond_endpoint_radius_scale
    )
    radii = np.asarray(scene.atom_radii, dtype=np.float32)
    source_offset = np.minimum(radii[pairs[:, 0]] * endpoint_scale, limit)
    other_offset = np.minimum(radii[pairs[:, 1]] * endpoint_scale, limit)
    start = source + direction * source_offset[:, None]
    end = source + delta - direction * other_offset[:, None]
    return (
        np.ascontiguousarray(start, dtype=np.float32),
        np.ascontiguousarray(end, dtype=np.float32),
    )


def _atom_style_id(color: np.ndarray, selected: bool) -> str:
    red, green, blue = (int(channel) for channel in color)
    return f"atom-{red:02x}{green:02x}{blue:02x}-{'s' if selected else 'n'}"


def _bond_style_id(color: np.ndarray) -> str:
    red, green, blue = (int(channel) for channel in color)
    return f"bond-{red:02x}{green:02x}{blue:02x}"


def _write_header(
    handle: TextIO,
    scene: VectorSceneSnapshot,
    atom_rgb: np.ndarray,
    selected: np.ndarray,
    bond_rgb: np.ndarray,
) -> None:
    width = int(scene.width)
    height = int(scene.height)
    background = _color_hex(scene.background)
    handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    handle.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
    )
    handle.write("<title>TrajPlayer molecular viewport</title>\n")
    handle.write(
        "<desc>Editable vector projection of the current TrajPlayer molecular view.</desc>\n"
    )
    handle.write("<defs>\n")
    handle.write(
        f'<clipPath id="viewport-clip"><rect width="{width}" height="{height}"/></clipPath>\n'
    )
    selection = np.clip(np.asarray(scene.selection_color, dtype=np.float64), 0.0, 1.0)
    atom_styles = np.unique(
        np.column_stack((atom_rgb, selected.astype(np.uint8))),
        axis=0,
    )
    for red, green, blue, flag_value in atom_styles:
        color8 = np.array([red, green, blue], dtype=np.uint8)
        flag = bool(flag_value)
        style_id = _atom_style_id(color8, flag)
        base = color8.astype(np.float64) / 255.0
        stops = _atom_gradient_stops(base, selection if flag else None)
        handle.write(
            f'<radialGradient id="{style_id}" cx="62%" cy="31%" r="72%" fx="58%" fy="26%">\n'
        )
        for offset, color in stops:
            handle.write(
                f'<stop offset="{offset}" stop-color="{_color_hex(color)}"/>\n'
            )
        handle.write("</radialGradient>\n")
    bond_styles = np.unique(bond_rgb.reshape(-1, 3), axis=0)
    for color8 in bond_styles:
        style_id = _bond_style_id(color8)
        base = color8.astype(np.float64) / 255.0
        handle.write(
            f'<linearGradient id="{style_id}" x1="0" y1="0" x2="0" y2="1">\n'
        )
        for offset, factor, lift in (
            ("0%", 0.38, 0.00),
            ("19%", 0.74, 0.02),
            ("50%", 1.00, 0.11),
            ("81%", 0.74, 0.02),
            ("100%", 0.38, 0.00),
        ):
            shaded = np.clip(base * factor + lift, 0.0, 1.0)
            handle.write(
                f'<stop offset="{offset}" stop-color="{_color_hex(shaded)}"/>\n'
            )
        handle.write("</linearGradient>\n")
    handle.write("</defs>\n")
    handle.write(
        f'<rect class="background" width="{width}" height="{height}" fill="{background}"/>\n'
    )


def _atom_gradient_stops(
    base: np.ndarray,
    selection: np.ndarray | None,
) -> tuple[tuple[str, np.ndarray], ...]:
    values = (
        ("0%", np.clip(base * 0.82 + 0.12, 0.0, 1.0), 0.34),
        ("42%", np.clip(base * 0.94 + 0.035, 0.0, 1.0), 0.34),
        ("74%", np.clip(base * 0.68 + 0.018, 0.0, 1.0), 0.38),
        ("91%", np.clip(base * 0.36 + 0.012, 0.0, 1.0), 0.55),
        ("100%", np.full(3, 0.030, dtype=np.float64), 0.72),
    )
    if selection is None:
        return tuple((offset, color) for offset, color, _mix in values)
    return tuple(
        (offset, color * (1.0 - mix) + selection * mix)
        for offset, color, mix in values
    )


def _write_atom(
    handle: TextIO,
    center: np.ndarray,
    rx: float,
    ry: float,
    style_id: str,
    atom_index: int,
) -> None:
    handle.write(
        f'<ellipse class="atom" data-atom-index="{atom_index}" '
        f'cx="{_number(center[0])}" cy="{_number(center[1])}" '
        f'rx="{_number(rx)}" ry="{_number(ry)}" fill="url(#{style_id})" '
        f'stroke="#151515" stroke-width="{_number(max(0.45, min(rx, ry) * 0.09))}"/>\n'
    )


def _write_bond(
    handle: TextIO,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    style_ids: tuple[str, str],
) -> None:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-4 or radius <= 0.0:
        return
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    half = length * 0.5
    overlap = min(0.25, half * 0.02)
    handle.write(
        f'<g class="bond" transform="translate({_number(start[0])} {_number(start[1])}) '
        f'rotate({_number(angle)})">\n'
    )
    handle.write(
        f'<rect x="0" y="{_number(-radius)}" width="{_number(half + overlap)}" '
        f'height="{_number(radius * 2.0)}" fill="url(#{style_ids[0]})"/>\n'
    )
    handle.write(
        f'<rect x="{_number(half - overlap)}" y="{_number(-radius)}" '
        f'width="{_number(half + overlap)}" height="{_number(radius * 2.0)}" '
        f'fill="url(#{style_ids[1]})"/>\n'
    )
    handle.write("</g>\n")


def _write_box_edge(
    handle: TextIO,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float],
) -> None:
    handle.write(
        f'<line class="periodic-box" x1="{_number(start[0])}" y1="{_number(start[1])}" '
        f'x2="{_number(end[0])}" y2="{_number(end[1])}" '
        f'stroke="{_color_hex(color)}" stroke-width="0.85" fill="none"/>\n'
    )


def _rgb8_array(colors: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    return np.floor(values * 255.0 + 0.5).astype(np.uint8)


def _color_hex(color: np.ndarray | tuple[float, float, float]) -> str:
    values = _rgb8_array(np.asarray(color, dtype=np.float64).reshape(1, 3))[0]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _number(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "0"
    return f"{number:.3f}".rstrip("0").rstrip(".")
