from __future__ import annotations

import math
import time

import numpy as np
from shiboken6 import VoidPtr
from PySide6.QtCore import QByteArray, QPoint, Qt
from PySide6.QtGui import QMatrix4x4, QMouseEvent, QSurfaceFormat, QVector3D, QWheelEvent
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .element_style import atom_render_arrays
from .render_stats import RenderStats


GL_FLOAT = 0x1406
GL_LINES = 0x0001
GL_TRIANGLE_STRIP = 0x0005
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_BUFFER = 0x8C2A
GL_R32F = 0x822E
GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02
POSITION_BUFFER_COUNT = 3

DEFAULT_BACKGROUND = (1.0, 1.0, 1.0, 1.0)
BALL_STICK_ATOM_RADIUS_SCALE = 0.25
BALL_ATOM_RADIUS_SCALE = 1.0
ATOM_RADIUS_SCALE = BALL_STICK_ATOM_RADIUS_SCALE
CAMERA_DISTANCE_SCALE = 2.75
MIN_ATOM_RADIUS_NDC_X = 1.0e-6
BOND_RADIUS_SCALE = 0.20
MIN_BOND_RADIUS_NDC_X = 1.0e-6
BOND_ELEMENT_COLOR_MIX = 0.94
BOND_NEUTRAL_COLOR = (0.40, 0.40, 0.38)
BOND_ENDPOINT_RADIUS_SCALE = 0.92
BOND_MAX_ENDPOINT_LENGTH_FRACTION = 0.45
BOX_COLOR = (0.18, 0.18, 0.18)
RENDER_MODE_BALL_STICK = "ball_stick"
RENDER_MODE_BALL = "ball"
RENDER_MODE_BOND = "bond"
RENDER_MODES = frozenset((RENDER_MODE_BALL_STICK, RENDER_MODE_BALL, RENDER_MODE_BOND))
BOX_EDGE_INDICES = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [1, 5],
        [2, 4],
        [2, 6],
        [3, 5],
        [3, 6],
        [4, 7],
        [5, 7],
        [6, 7],
    ],
    dtype=np.int32,
)


def framebuffer_pixel_size(logical_width: int, logical_height: int, device_pixel_ratio: float) -> tuple[int, int]:
    scale = max(float(device_pixel_ratio), 1.0e-6)
    return (
        max(1, int(math.floor(float(logical_width) * scale + 0.5))),
        max(1, int(math.floor(float(logical_height) * scale + 0.5))),
    )


def atom_radius_scale_for_mode(mode: str, user_scale: float) -> float:
    base_scale = BALL_ATOM_RADIUS_SCALE if mode == RENDER_MODE_BALL else BALL_STICK_ATOM_RADIUS_SCALE
    return float(base_scale * float(user_scale))


def bond_segment_colors_for_pairs(atom_colors: np.ndarray, bond_pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(bond_pairs, dtype=np.int32)
    if pairs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("bond_pairs must have shape (M, 2)")

    colors = np.asarray(atom_colors, dtype=np.float32)
    segment_colors = np.empty((pairs.shape[0] * 2, 3), dtype=np.float32)
    segment_colors[0::2] = colors[pairs[:, 0]]
    segment_colors[1::2] = colors[pairs[:, 1]]

    neutral = np.asarray(BOND_NEUTRAL_COLOR, dtype=np.float32)
    segment_colors *= np.float32(BOND_ELEMENT_COLOR_MIX)
    segment_colors += neutral * np.float32(1.0 - BOND_ELEMENT_COLOR_MIX)
    np.clip(segment_colors, 0.0, 1.0, out=segment_colors)
    return np.ascontiguousarray(segment_colors, dtype=np.float32)


def cell_box_vertices(cell: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cell, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    a, b, c = matrix[0], matrix[1], matrix[2]
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            a,
            b,
            c,
            a + b,
            a + c,
            b + c,
            a + b + c,
        ],
        dtype=np.float32,
    )
    vertices = corners[BOX_EDGE_INDICES.reshape(-1)]
    return np.ascontiguousarray(vertices, dtype=np.float32)


def periodic_cell_inverse_columns(cell: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        return None
    if np.min(np.linalg.norm(matrix, axis=1)) <= 1.0e-8:
        return None
    if abs(float(np.linalg.det(matrix))) <= 1.0e-8:
        return None
    inverse = np.linalg.inv(matrix)
    return np.ascontiguousarray(inverse.T, dtype=np.float32)


def minimum_image_displacements(
    displacements: np.ndarray,
    cell: np.ndarray,
    *,
    inverse_columns: np.ndarray | None = None,
) -> np.ndarray:
    delta = np.asarray(displacements, dtype=np.float32)
    if delta.ndim < 1 or delta.shape[-1] != 3:
        raise ValueError("displacements must end with three Cartesian coordinates")
    matrix = np.asarray(cell, dtype=np.float32)
    inverse = (
        periodic_cell_inverse_columns(matrix)
        if inverse_columns is None
        else np.asarray(inverse_columns, dtype=np.float32)
    )
    if inverse is None:
        return np.ascontiguousarray(delta, dtype=np.float32)
    if inverse.shape != (3, 3):
        raise ValueError("inverse_columns must have shape (3, 3)")
    fractional = delta @ inverse.T
    fractional -= np.rint(fractional)
    wrapped = fractional @ matrix
    return np.ascontiguousarray(wrapped, dtype=np.float32)


def periodic_anchor_index(
    positions: np.ndarray,
    atom_indices: np.ndarray,
    cell: np.ndarray,
    *,
    inverse_columns: np.ndarray | None = None,
) -> int:
    frame = np.asarray(positions, dtype=np.float32)
    indices = np.asarray(atom_indices, dtype=np.int32)
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("atom_indices must be a non-empty 1D array")
    matrix = np.asarray(cell, dtype=np.float32)
    inverse = (
        periodic_cell_inverse_columns(matrix)
        if inverse_columns is None
        else np.asarray(inverse_columns, dtype=np.float32)
    )
    if inverse is None:
        return int(indices[0])
    if inverse.shape != (3, 3):
        raise ValueError("inverse_columns must have shape (3, 3)")

    fractional = frame[indices] @ inverse.T
    angles = fractional * np.float32(2.0 * math.pi)
    circular_center = np.arctan2(
        np.mean(np.sin(angles), axis=0),
        np.mean(np.cos(angles), axis=0),
    ) / np.float32(2.0 * math.pi)
    relative_fractional = fractional - circular_center.astype(np.float32)
    relative_fractional -= np.rint(relative_fractional)
    relative_cartesian = relative_fractional @ matrix
    distances2 = np.einsum(
        "ij,ij->i",
        relative_cartesian,
        relative_cartesian,
        dtype=np.float32,
    )
    return int(indices[int(np.argmin(distances2))])


def unwrap_positions_by_anchor_indices(
    positions: np.ndarray,
    atom_indices: np.ndarray,
    anchor_indices: np.ndarray,
    cell: np.ndarray,
    *,
    inverse_columns: np.ndarray | None = None,
) -> np.ndarray:
    frame = np.asarray(positions, dtype=np.float32)
    indices = np.asarray(atom_indices, dtype=np.int32)
    anchors = np.asarray(anchor_indices, dtype=np.int32)
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("atom_indices must be a non-empty 1D array")
    if anchors.ndim != 1 or anchors.shape != indices.shape:
        raise ValueError("anchor_indices must match atom_indices")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= frame.shape[0]):
        raise ValueError("atom_indices contains an index outside the position array")
    if anchors.size and int(anchors.max()) >= frame.shape[0]:
        raise ValueError("anchor_indices contains an index outside the position array")

    displayed = np.ascontiguousarray(frame[indices], dtype=np.float32)
    anchored = anchors >= 0
    if not np.any(anchored):
        return displayed
    current_anchors = frame[anchors[anchored]]
    displacements = displayed[anchored] - current_anchors
    displayed[anchored] = current_anchors + minimum_image_displacements(
        displacements,
        cell,
        inverse_columns=inverse_columns,
    )
    return displayed


def bond_segment_endpoints_for_frame(
    positions: np.ndarray, bond_pairs: np.ndarray, atom_radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(bond_pairs, dtype=np.int32)
    if pairs.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("bond_pairs must have shape (M, 2)")

    frame = np.asarray(positions, dtype=np.float32)
    radii = np.asarray(atom_radii, dtype=np.float32)
    left = frame[pairs[:, 0]]
    right = frame[pairs[:, 1]]
    delta = right - left
    lengths = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))
    safe_lengths = np.maximum(lengths, np.float32(1.0e-6))
    direction = delta / safe_lengths[:, None]
    limit = lengths * np.float32(BOND_MAX_ENDPOINT_LENGTH_FRACTION)
    radius_scale = np.float32(ATOM_RADIUS_SCALE * BOND_ENDPOINT_RADIUS_SCALE)
    left_offsets = np.minimum(radii[pairs[:, 0]] * radius_scale, limit)
    right_offsets = np.minimum(radii[pairs[:, 1]] * radius_scale, limit)
    mid = (left + right) * np.float32(0.5)

    starts = np.empty((pairs.shape[0] * 2, 3), dtype=np.float32)
    ends = np.empty((pairs.shape[0] * 2, 3), dtype=np.float32)
    starts[0::2] = left + direction * left_offsets[:, None]
    ends[0::2] = mid
    starts[1::2] = right - direction * right_offsets[:, None]
    ends[1::2] = mid
    return np.ascontiguousarray(starts, dtype=np.float32), np.ascontiguousarray(ends, dtype=np.float32)


VERTEX_SHADER = f"""
#version 330 core
layout(location = 1) in float a_atom_index;
layout(location = 2) in float a_radius;
layout(location = 3) in vec3 a_color;
layout(location = 8) in float a_unwrap_anchor_index;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform samplerBuffer u_positions;
uniform float u_atom_size_scale;
uniform int u_has_periodic_cell;
uniform vec3 u_cell_a;
uniform vec3 u_cell_b;
uniform vec3 u_cell_c;
uniform vec3 u_inv_cell_0;
uniform vec3 u_inv_cell_1;
uniform vec3 u_inv_cell_2;

const float MIN_RADIUS_NDC_X = {MIN_ATOM_RADIUS_NDC_X:.8f};

out vec2 v_corner;
out vec3 v_color;
out vec3 v_view_center;
out float v_radius;

vec3 position_at(int atom_index) {{
    int offset = atom_index * 3;
    return vec3(
        texelFetch(u_positions, offset).r,
        texelFetch(u_positions, offset + 1).r,
        texelFetch(u_positions, offset + 2).r
    );
}}

vec3 minimum_image_delta(vec3 delta) {{
    vec3 fractional = vec3(
        dot(delta, u_inv_cell_0),
        dot(delta, u_inv_cell_1),
        dot(delta, u_inv_cell_2)
    );
    fractional -= round(fractional);
    return u_cell_a * fractional.x + u_cell_b * fractional.y + u_cell_c * fractional.z;
}}

vec3 display_position(int atom_index, int anchor_index) {{
    vec3 position = position_at(atom_index);
    if (u_has_periodic_cell == 0 || anchor_index < 0) {{
        return position;
    }}
    vec3 anchor = position_at(anchor_index);
    return anchor + minimum_image_delta(position - anchor);
}}

vec2 corner_from_vertex_id() {{
    int id = gl_VertexID & 3;
    float x = (id == 1 || id == 3) ? 1.0 : -1.0;
    float y = (id >= 2) ? 1.0 : -1.0;
    return vec2(x, y);
}}

void main() {{
    vec2 corner = corner_from_vertex_id();
    vec3 atom_position = display_position(
        int(a_atom_index + 0.5),
        int(a_unwrap_anchor_index)
    );
    vec4 center = u_view * vec4(atom_position, 1.0);
    float radius = max(a_radius * u_atom_size_scale, 0.001);
    vec4 clip_center = u_proj * center;
    vec4 clip_edge = u_proj * vec4(center.xyz + vec3(radius, 0.0, 0.0), 1.0);
    float aspect = u_proj[1][1] / max(u_proj[0][0], 0.0001);
    float radius_ndc_x = max(abs((clip_edge.x / clip_edge.w) - (clip_center.x / clip_center.w)), MIN_RADIUS_NDC_X);
    vec2 ndc_offset = corner * vec2(radius_ndc_x, radius_ndc_x * aspect);
    gl_Position = clip_center;
    gl_Position.xy += ndc_offset * gl_Position.w;
    v_corner = corner;
    v_color = a_color;
    v_view_center = center.xyz;
    v_radius = radius;
}}
"""


FRAGMENT_SHADER_BODY = """
uniform mat4 u_proj;

in vec2 v_corner;
in vec3 v_color;
in vec3 v_view_center;
in float v_radius;
out vec4 frag_color;

void main() {
    float r2 = dot(v_corner, v_corner);
    if (r2 > 1.0) {
        discard;
    }
    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 normal = normalize(vec3(v_corner, z));
    vec3 surface_view = v_view_center + vec3(v_corner * v_radius, z * v_radius);
    vec4 surface_clip = u_proj * vec4(surface_view, 1.0);
    gl_FragDepth = clamp(surface_clip.z / surface_clip.w * 0.5 + 0.5, 0.0, 1.0);
    vec3 light = normalize(vec3(0.35, 0.55, 0.76));
    float diffuse = max(dot(normal, light), 0.0);
    float edge_shadow = 0.62 + 0.38 * smoothstep(0.12, 0.92, z);
    float atom_outline = smoothstep(0.82, 0.99, sqrt(r2));
    float rim_shadow = 1.0 - 0.42 * atom_outline;
    float z2 = z * z;
    float z4 = z2 * z2;
    float z8 = z4 * z4;
    float z14 = z8 * z4 * z2;
    vec3 color = (v_color * edge_shadow * (0.20 + 0.80 * diffuse) + vec3(0.12) * z14) * rim_shadow;
    color = mix(color, vec3(0.045), 0.34 * atom_outline);
    frag_color = vec4(color, 1.0);
}
"""


FRAGMENT_SHADER = "#version 330 core\n" + FRAGMENT_SHADER_BODY
CONSERVATIVE_FRAGMENT_SHADER = (
    "#version 330 core\n"
    "#extension GL_ARB_conservative_depth : require\n"
    "layout(depth_less) out float gl_FragDepth;\n"
    + FRAGMENT_SHADER_BODY
)


BOND_VERTEX_SHADER = f"""
#version 330 core
layout(location = 4) in vec4 a_bond_data;
layout(location = 5) in vec3 a_bond_color_a;
layout(location = 6) in vec3 a_bond_color_b;
layout(location = 8) in float a_unwrap_anchor_index;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform samplerBuffer u_positions;
uniform float u_atom_size_scale;
uniform float u_bond_size_scale;
uniform float u_endpoint_radius_scale;
uniform int u_has_periodic_cell;
uniform vec3 u_cell_a;
uniform vec3 u_cell_b;
uniform vec3 u_cell_c;
uniform vec3 u_inv_cell_0;
uniform vec3 u_inv_cell_1;
uniform vec3 u_inv_cell_2;

const float BOND_RADIUS_SCALE = {BOND_RADIUS_SCALE:.8f};
const float MIN_BOND_RADIUS_NDC_X = {MIN_BOND_RADIUS_NDC_X:.8f};
const float MAX_ENDPOINT_LENGTH_FRACTION = {BOND_MAX_ENDPOINT_LENGTH_FRACTION:.8f};

out float v_side;
out float v_along;
out vec3 v_color_a;
out vec3 v_color_b;

vec3 position_at(int atom_index) {{
    int offset = atom_index * 3;
    return vec3(
        texelFetch(u_positions, offset).r,
        texelFetch(u_positions, offset + 1).r,
        texelFetch(u_positions, offset + 2).r
    );
}}

vec3 minimum_image_delta(vec3 delta) {{
    if (u_has_periodic_cell == 0) {{
        return delta;
    }}
    vec3 fractional = vec3(
        dot(delta, u_inv_cell_0),
        dot(delta, u_inv_cell_1),
        dot(delta, u_inv_cell_2)
    );
    fractional -= round(fractional);
    return u_cell_a * fractional.x + u_cell_b * fractional.y + u_cell_c * fractional.z;
}}

vec3 display_position(int atom_index, int anchor_index) {{
    vec3 position = position_at(atom_index);
    if (u_has_periodic_cell == 0 || anchor_index < 0) {{
        return position;
    }}
    vec3 anchor = position_at(anchor_index);
    return anchor + minimum_image_delta(position - anchor);
}}

void main() {{
    int id = gl_VertexID & 3;
    float along = (id == 1 || id == 3) ? 1.0 : 0.0;
    float side = (id >= 2) ? 1.0 : -1.0;

    int anchor_index = int(a_unwrap_anchor_index);
    vec3 source = display_position(int(a_bond_data.x + 0.5), anchor_index);
    vec3 other = display_position(int(a_bond_data.y + 0.5), anchor_index);
    vec3 delta = minimum_image_delta(other - source);
    other = source + delta;
    float bond_length = length(delta);
    vec3 direction = bond_length > 0.000001 ? delta / bond_length : vec3(1.0, 0.0, 0.0);
    float source_offset = min(
        a_bond_data.z * u_atom_size_scale * u_endpoint_radius_scale,
        bond_length * MAX_ENDPOINT_LENGTH_FRACTION
    );
    float other_offset = min(
        a_bond_data.w * u_atom_size_scale * u_endpoint_radius_scale,
        bond_length * MAX_ENDPOINT_LENGTH_FRACTION
    );
    vec3 bond_start = source + direction * source_offset;
    vec3 bond_end = other - direction * other_offset;

    vec4 view_a = u_view * vec4(bond_start, 1.0);
    vec4 view_b = u_view * vec4(bond_end, 1.0);
    vec4 clip_a = u_proj * view_a;
    vec4 clip_b = u_proj * view_b;
    vec2 ndc_a = clip_a.xy / clip_a.w;
    vec2 ndc_b = clip_b.xy / clip_b.w;
    vec2 axis = ndc_b - ndc_a;
    float axis_len = length(axis);
    vec2 perp = axis_len > 0.00001 ? vec2(-axis.y, axis.x) / axis_len : vec2(0.0, 1.0);

    vec4 view_mid = mix(view_a, view_b, 0.5);
    vec4 clip_mid = u_proj * view_mid;
    vec4 clip_edge = u_proj * vec4(
        view_mid.xyz + vec3(BOND_RADIUS_SCALE * u_bond_size_scale, 0.0, 0.0),
        1.0
    );
    float aspect = u_proj[1][1] / max(u_proj[0][0], 0.0001);
    float radius_ndc_x = max(abs((clip_edge.x / clip_edge.w) - (clip_mid.x / clip_mid.w)), MIN_BOND_RADIUS_NDC_X);
    vec2 offset = perp * side * radius_ndc_x * vec2(1.0, aspect);

    vec4 clip = mix(clip_a, clip_b, along);
    gl_Position = clip;
    gl_Position.xy += offset * gl_Position.w;
    v_side = side;
    v_along = along;
    v_color_a = a_bond_color_a;
    v_color_b = a_bond_color_b;
}}
"""


BOND_FRAGMENT_SHADER = """
#version 330 core
in float v_side;
in float v_along;
in vec3 v_color_a;
in vec3 v_color_b;
out vec4 frag_color;

void main() {
    float side = clamp(abs(v_side), 0.0, 1.0);
    float roundness = sqrt(max(0.0, 1.0 - side * side));
    float diffuse = 0.38 + 0.62 * roundness;
    float roundness2 = roundness * roundness;
    float roundness4 = roundness2 * roundness2;
    vec3 highlight = vec3(0.11) * roundness4 * roundness4;
    vec3 bond_color = v_along < 0.5 ? v_color_a : v_color_b;
    frag_color = vec4(bond_color * diffuse + highlight, 1.0);
}
"""


BOX_VERTEX_SHADER = """
#version 330 core
layout(location = 7) in vec3 a_box_position;

uniform mat4 u_view;
uniform mat4 u_proj;

void main() {
    gl_Position = u_proj * u_view * vec4(a_box_position, 1.0);
}
"""


BOX_FRAGMENT_SHADER = f"""
#version 330 core
out vec4 frag_color;

void main() {{
    frag_color = vec4({BOX_COLOR[0]:.8f}, {BOX_COLOR[1]:.8f}, {BOX_COLOR[2]:.8f}, 1.0);
}}
"""


def default_surface_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(0)
    return fmt


class MoleculeGLWidget(QOpenGLWidget):
    """GPU instanced molecule view; Qt is only the host surface."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._program: QOpenGLShaderProgram | None = None
        self._bond_program: QOpenGLShaderProgram | None = None
        self._box_program: QOpenGLShaderProgram | None = None
        self._vao = QOpenGLVertexArrayObject()
        self._bond_vao = QOpenGLVertexArrayObject()
        self._box_vao = QOpenGLVertexArrayObject()
        self._quad_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._position_vbos = [
            QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            for _ in range(POSITION_BUFFER_COUNT)
        ]
        self._position_vbo = self._position_vbos[0]
        self._atom_index_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._radius_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._color_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._atom_unwrap_anchor_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._bond_data_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._bond_color_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._box_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._position_textures: list[QOpenGLTexture] = []
        self._position_texture: QOpenGLTexture | None = None
        self._position_buffer_index = -1
        self._gl = None
        self._loc_view = -1
        self._loc_proj = -1
        self._loc_positions = -1
        self._loc_atom_size_scale = -1
        self._loc_has_periodic_cell = -1
        self._loc_cell_vectors = (-1, -1, -1)
        self._loc_inverse_columns = (-1, -1, -1)
        self._bond_loc_view = -1
        self._bond_loc_proj = -1
        self._bond_loc_positions = -1
        self._bond_loc_atom_size_scale = -1
        self._bond_loc_bond_size_scale = -1
        self._bond_loc_endpoint_radius_scale = -1
        self._bond_loc_has_periodic_cell = -1
        self._bond_loc_cell_vectors = (-1, -1, -1)
        self._bond_loc_inverse_columns = (-1, -1, -1)
        self._box_loc_view = -1
        self._box_loc_proj = -1
        self._viewport_size = (1, 1)

        self._atom_count = 0
        self._positions = np.empty((0, 3), dtype=np.float32)
        self._covalent_radii = np.empty((0,), dtype=np.float32)
        self._vdw_radii = np.empty((0,), dtype=np.float32)
        self._radii = np.empty((0,), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.float32)
        self._visible_atom_indices = np.empty((0,), dtype=np.int32)
        self._render_atom_indices = np.empty((0,), dtype=np.int32)
        self._visible_atom_indices_gpu = np.empty((0,), dtype=np.float32)
        self._atom_unwrap_anchor_indices = np.empty((0,), dtype=np.int32)
        self._visible_unwrap_anchor_indices_gpu = np.empty((0,), dtype=np.float32)
        self._visible_radii = np.empty((0,), dtype=np.float32)
        self._visible_colors = np.empty((0, 3), dtype=np.float32)
        self._atom_visible_mask = np.empty((0,), dtype=np.bool_)
        self._visible_atom_count = 0
        self._depth_order_dirty = True
        self._positions_uploaded = False
        self._frame_upload_pending = True
        self._box_enabled = True
        self._box_has_cell = False
        self._periodic_cell_valid = False
        self._box_vertex_count = 0
        self._box_cell = np.empty((0, 3), dtype=np.float32)
        self._cell_inverse_columns = np.empty((0, 3), dtype=np.float32)
        self._box_vertices = np.empty((0, 3), dtype=np.float32)
        self._box_buffer_dirty = True
        self._bond_pairs = np.empty((0, 2), dtype=np.int32)
        self._bond_instance_count = 0
        self._bond_instance_data = np.empty((0, 5), dtype=np.float32)
        self._bond_instance_colors = np.empty((0, 6), dtype=np.float32)
        self._render_mode = RENDER_MODE_BALL_STICK
        self._show_atoms = True
        self._show_bonds = True
        self._atom_size_scale = 1.0
        self._bond_size_scale = 1.0

        self._center = QVector3D(0.0, 0.0, 0.0)
        self._scene_radius = 10.0
        self._yaw = 25.0
        self._pitch = -18.0
        self._distance = 40.0
        self._last_mouse_pos: QPoint | None = None
        self._background = DEFAULT_BACKGROUND
        self._benchmark_finish_gpu = False
        self._render_stats: RenderStats | None = None
        self._last_upload_ms = 0.0
        self._immediate_paint = False
        self._cleaning_up = False
        self._conservative_depth_enabled = False
        self._gl_vendor = "unknown"
        self._gl_renderer = "unknown"
        self._gl_version = "unknown"

    @property
    def render_stats(self) -> RenderStats | None:
        return self._render_stats

    @property
    def conservative_depth_enabled(self) -> bool:
        return self._conservative_depth_enabled

    @property
    def gl_diagnostics(self) -> dict[str, str]:
        return {
            "vendor": self._gl_vendor,
            "renderer": self._gl_renderer,
            "version": self._gl_version,
        }

    def enable_benchmark_stats(self, *, finish_gpu: bool) -> None:
        self._benchmark_finish_gpu = bool(finish_gpu)
        self._render_stats = RenderStats()
        self._immediate_paint = True

    def set_immediate_paint(self, enabled: bool) -> None:
        self._immediate_paint = bool(enabled)

    def set_atoms(self, atom_numbers: np.ndarray) -> None:
        covalent_radii, vdw_radii, colors = atom_render_arrays(atom_numbers)
        self._atom_count = int(covalent_radii.shape[0])
        self._covalent_radii = covalent_radii
        self._vdw_radii = vdw_radii
        self._radii = self._radii_for_render_mode(self._render_mode)
        self._colors = colors
        self._positions = np.zeros((self._atom_count, 3), dtype=np.float32)
        self._atom_unwrap_anchor_indices = np.full(
            self._atom_count,
            -1,
            dtype=np.int32,
        )
        self._set_visible_atom_arrays(np.arange(self._atom_count, dtype=np.int32))
        self._positions_uploaded = False
        self._frame_upload_pending = True
        self._reset_bonds()
        self.set_cell(None)
        if self.isValid():
            self.makeCurrent()
            self._upload_static_buffers()
            self._allocate_position_buffer()
            self._upload_bond_static_buffers()
            self._upload_box_buffer()
            self.doneCurrent()
        self.update()

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @property
    def atom_size_scale(self) -> float:
        return self._atom_size_scale

    @property
    def bond_size_scale(self) -> float:
        return self._bond_size_scale

    @property
    def effective_atom_size_scale(self) -> float:
        return atom_radius_scale_for_mode(self._render_mode, self._atom_size_scale)

    @property
    def bond_count(self) -> int:
        return self._bond_instance_count

    def set_render_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in RENDER_MODES:
            raise ValueError(f"Unsupported render mode: {mode}")
        if normalized == self._render_mode:
            self.update()
            return

        self._render_mode = normalized
        self._show_atoms = normalized != RENDER_MODE_BOND
        self._show_bonds = normalized != RENDER_MODE_BALL
        new_radii = self._radii_for_render_mode(normalized)
        radii_changed = new_radii is not self._radii
        self._radii = new_radii
        if radii_changed and self._atom_count > 0:
            self._set_visible_atom_arrays(self._visible_atom_indices)
            self._rebuild_visible_bonds()
            if self.isValid():
                self.makeCurrent()
                self._upload_static_buffers()
                self._upload_bond_static_buffers()
                self.doneCurrent()
        self.update()

    def set_atom_size_scale(self, scale: float) -> None:
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Atom size scale must be a finite positive value")
        self._atom_size_scale = value
        self.update()

    def set_bond_size_scale(self, scale: float) -> None:
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Bond size scale must be a finite positive value")
        self._bond_size_scale = value
        self.update()

    def set_visible_atoms(
        self,
        atom_indices: np.ndarray | None,
        *,
        fit_view: bool = True,
        unwrap_periodic: bool = False,
        unwrap_group_ids: np.ndarray | None = None,
    ) -> None:
        if atom_indices is None:
            indices = np.arange(self._atom_count, dtype=np.int32)
        else:
            indices = np.asarray(atom_indices, dtype=np.int32)
            if indices.ndim != 1:
                raise ValueError("atom_indices must be a 1D array")
            if indices.size == 0:
                raise ValueError("atom_indices must contain at least one atom")
            if int(indices.min()) < 0 or int(indices.max()) >= self._atom_count:
                raise ValueError("atom_indices contains an index outside the current atom array")
            indices = np.unique(indices)
        self._atom_unwrap_anchor_indices = np.full(
            self._atom_count,
            -1,
            dtype=np.int32,
        )
        if unwrap_periodic:
            if unwrap_group_ids is None:
                selected_group_ids = np.zeros(indices.shape, dtype=np.int32)
            else:
                group_ids = np.asarray(unwrap_group_ids, dtype=np.int32)
                if group_ids.shape != (self._atom_count,):
                    raise ValueError(
                        "unwrap_group_ids must have one group ID for every atom"
                    )
                selected_group_ids = group_ids[indices]
            for group_id in np.unique(selected_group_ids):
                group_indices = indices[selected_group_ids == group_id]
                anchor_index = (
                    periodic_anchor_index(
                        self._positions,
                        group_indices,
                        self._box_cell,
                        inverse_columns=self._cell_inverse_columns,
                    )
                    if self._periodic_cell_valid
                    else int(group_indices[0])
                )
                self._atom_unwrap_anchor_indices[group_indices] = anchor_index
        self._set_visible_atom_arrays(np.ascontiguousarray(indices, dtype=np.int32))
        self._rebuild_visible_bonds()
        if fit_view and self._positions.size:
            self._fit_camera_to_frame(
                self._visible_positions(),
                include_box=self._visible_atom_count == self._atom_count,
                minimum_radius=1.8 if self._visible_atom_count < self._atom_count else 1.0,
            )
        if self.isValid():
            self.makeCurrent()
            self._upload_static_buffers()
            self._upload_bond_static_buffers()
            self.doneCurrent()
        self.update()

    def set_box_enabled(self, enabled: bool) -> None:
        self._box_enabled = bool(enabled)
        self._box_vertex_count = int(self._box_vertices.shape[0]) if self._box_enabled and self._box_has_cell else 0
        self._box_buffer_dirty = True
        self.update()

    def set_cell(self, cell: np.ndarray | None) -> None:
        if cell is None:
            if not self._box_has_cell:
                return
            self._box_has_cell = False
            self._periodic_cell_valid = False
            self._box_cell = np.empty((0, 3), dtype=np.float32)
            self._cell_inverse_columns = np.empty((0, 3), dtype=np.float32)
            self._box_vertices = np.empty((0, 3), dtype=np.float32)
        else:
            matrix = np.ascontiguousarray(cell, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError(f"Expected cell shape (3, 3), got {matrix.shape}")
            if self._box_has_cell and np.array_equal(matrix, self._box_cell):
                return
            self._box_has_cell = bool(np.any(matrix))
            self._box_cell = matrix
            inverse_columns = periodic_cell_inverse_columns(matrix)
            self._periodic_cell_valid = inverse_columns is not None
            self._cell_inverse_columns = (
                inverse_columns
                if inverse_columns is not None
                else np.empty((0, 3), dtype=np.float32)
            )
            self._box_vertices = cell_box_vertices(matrix) if self._box_has_cell else np.empty((0, 3), dtype=np.float32)
        self._box_vertex_count = int(self._box_vertices.shape[0]) if self._box_enabled and self._box_has_cell else 0
        self._box_buffer_dirty = True
        self.update()

    def set_bonds(self, bond_pairs: np.ndarray) -> None:
        pairs = np.asarray(bond_pairs, dtype=np.int32)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("bond_pairs must have shape (M, 2)")
        if pairs.size and (int(pairs.min()) < 0 or int(pairs.max()) >= self._atom_count):
            raise ValueError("bond_pairs contains atom indices outside the current atom array")

        self._bond_pairs = np.ascontiguousarray(pairs, dtype=np.int32)
        self._rebuild_visible_bonds()
        if self.isValid():
            self.makeCurrent()
            self._upload_bond_static_buffers()
            self.doneCurrent()
        self.update()

    def set_frame(self, positions: np.ndarray, *, reset_view: bool = False, cell: np.ndarray | None = None) -> None:
        frame = np.asarray(positions, dtype=np.float32)
        if frame.shape != (self._atom_count, 3):
            raise ValueError(f"Expected frame shape {(self._atom_count, 3)}, got {frame.shape}")
        self.set_cell(cell)
        if self._positions.shape != frame.shape:
            self._positions = np.empty(frame.shape, dtype=np.float32)
        np.copyto(self._positions, frame)
        self._frame_upload_pending = True
        if reset_view:
            self._fit_camera_to_frame(
                self._visible_positions(),
                include_box=self._visible_atom_count == self._atom_count,
                minimum_radius=1.8 if self._visible_atom_count < self._atom_count else 1.0,
            )
            self._depth_order_dirty = True
        if self._immediate_paint and self.isVisible():
            self.repaint()
        else:
            self.update()

    def set_background_rgb(self, rgb: tuple[float, float, float]) -> None:
        self._background = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        self.update()

    def reset_view(self) -> None:
        self._fit_camera_to_frame(
            self._visible_positions(),
            include_box=self._visible_atom_count == self._atom_count,
            minimum_radius=1.8 if self._visible_atom_count < self._atom_count else 1.0,
        )
        self.update()

    def initializeGL(self) -> None:  # type: ignore[override]
        self.context().aboutToBeDestroyed.connect(self.cleanup)
        self._gl = self.context().extraFunctions()
        self._gl.initializeOpenGLFunctions()
        self._gl.glEnable(GL_DEPTH_TEST)
        self._gl_vendor = self._read_gl_string(GL_VENDOR)
        self._gl_renderer = self._read_gl_string(GL_RENDERER)
        self._gl_version = self._read_gl_string(GL_VERSION)
        context_format = self.context().format()
        self._conservative_depth_enabled = (
            (context_format.majorVersion(), context_format.minorVersion()) >= (4, 2)
            or self.context().hasExtension(QByteArray(b"GL_ARB_conservative_depth"))
        )
        self._depth_order_dirty = True

        self._program = QOpenGLShaderProgram(self)
        if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER):
            raise RuntimeError(self._program.log())
        atom_fragment_shader = (
            CONSERVATIVE_FRAGMENT_SHADER
            if self._conservative_depth_enabled
            else FRAGMENT_SHADER
        )
        if not self._program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            atom_fragment_shader,
        ):
            raise RuntimeError(self._program.log())
        if not self._program.link():
            raise RuntimeError(self._program.log())
        self._loc_view = self._program.uniformLocation("u_view")
        self._loc_proj = self._program.uniformLocation("u_proj")
        self._loc_positions = self._program.uniformLocation("u_positions")
        self._loc_atom_size_scale = self._program.uniformLocation("u_atom_size_scale")
        self._loc_has_periodic_cell = self._program.uniformLocation(
            "u_has_periodic_cell"
        )
        self._loc_cell_vectors = tuple(
            self._program.uniformLocation(name)
            for name in ("u_cell_a", "u_cell_b", "u_cell_c")
        )
        self._loc_inverse_columns = tuple(
            self._program.uniformLocation(name)
            for name in ("u_inv_cell_0", "u_inv_cell_1", "u_inv_cell_2")
        )
        if min(
            self._loc_atom_size_scale,
            self._loc_has_periodic_cell,
            *self._loc_cell_vectors,
            *self._loc_inverse_columns,
        ) < 0:
            raise RuntimeError("Atom shader uniforms are unavailable")

        self._bond_program = QOpenGLShaderProgram(self)
        if not self._bond_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, BOND_VERTEX_SHADER):
            raise RuntimeError(self._bond_program.log())
        if not self._bond_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, BOND_FRAGMENT_SHADER):
            raise RuntimeError(self._bond_program.log())
        if not self._bond_program.link():
            raise RuntimeError(self._bond_program.log())
        self._bond_loc_view = self._bond_program.uniformLocation("u_view")
        self._bond_loc_proj = self._bond_program.uniformLocation("u_proj")
        self._bond_loc_positions = self._bond_program.uniformLocation("u_positions")
        self._bond_loc_atom_size_scale = self._bond_program.uniformLocation("u_atom_size_scale")
        self._bond_loc_bond_size_scale = self._bond_program.uniformLocation("u_bond_size_scale")
        self._bond_loc_endpoint_radius_scale = self._bond_program.uniformLocation(
            "u_endpoint_radius_scale"
        )
        self._bond_loc_has_periodic_cell = self._bond_program.uniformLocation(
            "u_has_periodic_cell"
        )
        self._bond_loc_cell_vectors = tuple(
            self._bond_program.uniformLocation(name)
            for name in ("u_cell_a", "u_cell_b", "u_cell_c")
        )
        self._bond_loc_inverse_columns = tuple(
            self._bond_program.uniformLocation(name)
            for name in ("u_inv_cell_0", "u_inv_cell_1", "u_inv_cell_2")
        )
        if min(
            self._bond_loc_atom_size_scale,
            self._bond_loc_bond_size_scale,
            self._bond_loc_endpoint_radius_scale,
            self._bond_loc_has_periodic_cell,
            *self._bond_loc_cell_vectors,
            *self._bond_loc_inverse_columns,
        ) < 0:
            raise RuntimeError("Bond shader uniforms are unavailable")

        self._box_program = QOpenGLShaderProgram(self)
        if not self._box_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, BOX_VERTEX_SHADER):
            raise RuntimeError(self._box_program.log())
        if not self._box_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, BOX_FRAGMENT_SHADER):
            raise RuntimeError(self._box_program.log())
        if not self._box_program.link():
            raise RuntimeError(self._box_program.log())
        self._box_loc_view = self._box_program.uniformLocation("u_view")
        self._box_loc_proj = self._box_program.uniformLocation("u_proj")

        self._vao.create()
        self._bond_vao.create()
        self._box_vao.create()
        self._quad_vbo.create()
        for buffer in self._position_vbos:
            buffer.create()
        self._atom_index_vbo.create()
        self._radius_vbo.create()
        self._color_vbo.create()
        self._atom_unwrap_anchor_vbo.create()
        self._bond_data_vbo.create()
        self._bond_color_vbo.create()
        self._box_vbo.create()
        self._position_textures = []
        for _ in range(POSITION_BUFFER_COUNT):
            texture = QOpenGLTexture(QOpenGLTexture.Target.TargetBuffer)
            if not texture.create():
                raise RuntimeError("Failed to create a GPU position buffer texture")
            self._position_textures.append(texture)

        self._upload_static_buffers()
        self._allocate_position_buffer()
        self._upload_bond_static_buffers()
        self._upload_box_buffer()
        self._upload_frame_buffers()

    def cleanup(self) -> None:
        if self._cleaning_up or not self.isValid():
            return
        self._cleaning_up = True
        try:
            self.makeCurrent()
            for texture in self._position_textures:
                if texture.isCreated():
                    texture.destroy()
            self._position_textures = []
            self._position_texture = None
            for buffer in (
                self._quad_vbo,
                self._atom_index_vbo,
                self._radius_vbo,
                self._color_vbo,
                self._atom_unwrap_anchor_vbo,
                self._bond_data_vbo,
                self._bond_color_vbo,
                self._box_vbo,
            ):
                if buffer.isCreated():
                    buffer.destroy()
            for buffer in self._position_vbos:
                if buffer.isCreated():
                    buffer.destroy()
            for vao in (self._vao, self._bond_vao, self._box_vao):
                if vao.isCreated():
                    vao.destroy()
            self.doneCurrent()
        finally:
            self._cleaning_up = False

    def paintGL(self) -> None:  # type: ignore[override]
        if self._gl is None or self._program is None:
            return

        draw_calls = 0
        self._last_upload_ms = 0.0
        if self._frame_upload_pending:
            self._upload_frame_buffers()
        if self._box_buffer_dirty:
            self._upload_box_buffer()
        if self._depth_order_dirty:
            self._rebuild_render_atom_arrays()
            self._upload_static_buffers()
        paint_start = time.perf_counter()
        self._set_physical_viewport()
        self._gl.glClearColor(*self._background)
        self._gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._atom_count <= 0:
            return

        view, proj = self._camera_matrices()
        if self._box_vertex_count > 0 and self._box_program is not None:
            self._box_program.bind()
            self._box_program.setUniformValue(self._box_loc_view, view)
            self._box_program.setUniformValue(self._box_loc_proj, proj)
            self._box_vao.bind()
            self._gl.glDrawArrays(GL_LINES, 0, self._box_vertex_count)
            draw_calls += 1
            self._box_vao.release()
            self._box_program.release()

        if self._show_atoms and self._visible_atom_count > 0:
            self._program.bind()
            self._program.setUniformValue(self._loc_view, view)
            self._program.setUniformValue(self._loc_proj, proj)
            self._program.setUniformValue(self._loc_positions, 0)
            self._gl.glUniform1f(self._loc_atom_size_scale, self.effective_atom_size_scale)
            self._set_periodic_uniforms(
                self._program,
                has_cell_location=self._loc_has_periodic_cell,
                cell_locations=self._loc_cell_vectors,
                inverse_locations=self._loc_inverse_columns,
            )
            self._gl.glActiveTexture(GL_TEXTURE0)
            self._gl.glBindTexture(GL_TEXTURE_BUFFER, self._position_texture.textureId())
            self._vao.bind()
            self._gl.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self._visible_atom_count)
            draw_calls += 1
            self._vao.release()
            self._gl.glBindTexture(GL_TEXTURE_BUFFER, 0)
            self._program.release()

        if self._show_bonds and self._bond_instance_count > 0 and self._bond_program is not None:
            self._bond_program.bind()
            self._bond_program.setUniformValue(self._bond_loc_view, view)
            self._bond_program.setUniformValue(self._bond_loc_proj, proj)
            self._bond_program.setUniformValue(self._bond_loc_positions, 0)
            self._gl.glUniform1f(
                self._bond_loc_atom_size_scale,
                self.effective_atom_size_scale,
            )
            self._gl.glUniform1f(self._bond_loc_bond_size_scale, self._bond_size_scale)
            endpoint_scale = BOND_ENDPOINT_RADIUS_SCALE if self._show_atoms else 0.0
            self._gl.glUniform1f(
                self._bond_loc_endpoint_radius_scale,
                endpoint_scale,
            )
            self._set_periodic_uniforms(
                self._bond_program,
                has_cell_location=self._bond_loc_has_periodic_cell,
                cell_locations=self._bond_loc_cell_vectors,
                inverse_locations=self._bond_loc_inverse_columns,
            )
            self._gl.glActiveTexture(GL_TEXTURE0)
            self._gl.glBindTexture(GL_TEXTURE_BUFFER, self._position_texture.textureId())
            self._bond_vao.bind()
            self._gl.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self._bond_instance_count)
            draw_calls += 1
            self._bond_vao.release()
            self._gl.glBindTexture(GL_TEXTURE_BUFFER, 0)
            self._bond_program.release()
        if self._benchmark_finish_gpu:
            self._gl.glFinish()
        if self._render_stats is not None:
            self._render_stats.record_frame(
                paint_ms=(time.perf_counter() - paint_start) * 1000.0,
                upload_ms=self._last_upload_ms,
                draw_calls=draw_calls,
                timestamp_s=paint_start,
            )

    def _set_periodic_uniforms(
        self,
        program: QOpenGLShaderProgram,
        *,
        has_cell_location: int,
        cell_locations: tuple[int, int, int],
        inverse_locations: tuple[int, int, int],
    ) -> None:
        has_periodic_cell = self._periodic_cell_valid
        self._gl.glUniform1i(has_cell_location, int(has_periodic_cell))
        if not has_periodic_cell:
            return
        for location, vector in zip(cell_locations, self._box_cell, strict=True):
            program.setUniformValue(
                location,
                QVector3D(float(vector[0]), float(vector[1]), float(vector[2])),
            )
        for location, vector in zip(
            inverse_locations,
            self._cell_inverse_columns,
            strict=True,
        ):
            program.setUniformValue(
                location,
                QVector3D(float(vector[0]), float(vector[1]), float(vector[2])),
            )

    def resizeGL(self, width: int, height: int) -> None:  # type: ignore[override]
        if self._gl is not None:
            self._set_physical_viewport()

    def _set_physical_viewport(self) -> None:
        width, height = framebuffer_pixel_size(self.width(), self.height(), self.devicePixelRatioF())
        self._viewport_size = (width, height)
        self._gl.glViewport(0, 0, width, height)

    def _read_gl_string(self, name: int) -> str:
        value = self._gl.glGetString(name)
        if value is None:
            return "unknown"
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._last_mouse_pos is None:
            self._last_mouse_pos = event.position().toPoint()
            return
        pos = event.position().toPoint()
        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._yaw += float(delta.x()) * 0.35
            self._pitch = max(-89.0, min(89.0, self._pitch + float(delta.y()) * 0.35))
            self._depth_order_dirty = True
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        steps = event.angleDelta().y() / 120.0
        self._distance *= math.pow(0.88, steps)
        self._distance = max(self._scene_radius * 0.6, min(self._scene_radius * 50.0, self._distance))
        self.update()

    def _upload_static_buffers(self) -> None:
        if self._program is None or not self._vao.isCreated():
            return
        self._program.bind()
        self._vao.bind()

        self._atom_index_vbo.bind()
        self._atom_index_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self._atom_index_vbo.allocate(
            self._visible_atom_indices_gpu.tobytes(),
            self._visible_atom_indices_gpu.nbytes,
        )
        self._program.enableAttributeArray(1)
        self._program.setAttributeBuffer(1, GL_FLOAT, 0, 1, 0)
        self._gl.glVertexAttribDivisor(1, 1)

        self._radius_vbo.bind()
        self._radius_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self._radius_vbo.allocate(self._visible_radii.tobytes(), self._visible_radii.nbytes)
        self._program.enableAttributeArray(2)
        self._program.setAttributeBuffer(2, GL_FLOAT, 0, 1, 0)
        self._gl.glVertexAttribDivisor(2, 1)

        self._color_vbo.bind()
        self._color_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self._color_vbo.allocate(self._visible_colors.tobytes(), self._visible_colors.nbytes)
        self._program.enableAttributeArray(3)
        self._program.setAttributeBuffer(3, GL_FLOAT, 0, 3, 0)
        self._gl.glVertexAttribDivisor(3, 1)

        self._atom_unwrap_anchor_vbo.bind()
        self._atom_unwrap_anchor_vbo.setUsagePattern(
            QOpenGLBuffer.UsagePattern.StaticDraw
        )
        self._atom_unwrap_anchor_vbo.allocate(
            self._visible_unwrap_anchor_indices_gpu.tobytes(),
            self._visible_unwrap_anchor_indices_gpu.nbytes,
        )
        self._program.enableAttributeArray(8)
        self._program.setAttributeBuffer(8, GL_FLOAT, 0, 1, 0)
        self._gl.glVertexAttribDivisor(8, 1)

        self._vao.release()
        self._program.release()

    def _allocate_position_buffer(self) -> None:
        if self._program is None or not self._vao.isCreated():
            return
        self._program.bind()
        self._vao.bind()
        allocation_bytes = max(1, self._atom_count * 3 * 4)
        for buffer, texture in zip(self._position_vbos, self._position_textures):
            buffer.bind()
            buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.StreamDraw)
            buffer.allocate(allocation_bytes)
            self._gl.glBindTexture(GL_TEXTURE_BUFFER, texture.textureId())
            self._gl.glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, buffer.bufferId())
        if self._position_textures:
            self._gl.glBindTexture(GL_TEXTURE_BUFFER, 0)
        self._vao.release()
        self._program.release()
        self._position_buffer_index = -1
        self._position_vbo = self._position_vbos[0]
        self._position_texture = None
        self._positions_uploaded = False
        self._frame_upload_pending = True

    def _upload_bond_static_buffers(self) -> None:
        if self._bond_program is None or not self._bond_vao.isCreated():
            return
        self._bond_program.bind()
        self._bond_vao.bind()

        self._bond_data_vbo.bind()
        self._bond_data_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self._bond_data_vbo.allocate(self._bond_instance_data.tobytes(), int(self._bond_instance_data.nbytes))
        self._bond_program.enableAttributeArray(4)
        self._bond_program.setAttributeBuffer(4, GL_FLOAT, 0, 4, 5 * 4)
        self._gl.glVertexAttribDivisor(4, 1)
        self._bond_program.enableAttributeArray(8)
        self._bond_program.setAttributeBuffer(8, GL_FLOAT, 4 * 4, 1, 5 * 4)
        self._gl.glVertexAttribDivisor(8, 1)

        self._bond_color_vbo.bind()
        self._bond_color_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        self._bond_color_vbo.allocate(
            self._bond_instance_colors.tobytes(),
            int(self._bond_instance_colors.nbytes),
        )
        self._bond_program.enableAttributeArray(5)
        self._bond_program.setAttributeBuffer(5, GL_FLOAT, 0, 3, 6 * 4)
        self._gl.glVertexAttribDivisor(5, 1)
        self._bond_program.enableAttributeArray(6)
        self._bond_program.setAttributeBuffer(6, GL_FLOAT, 3 * 4, 3, 6 * 4)
        self._gl.glVertexAttribDivisor(6, 1)

        self._bond_vao.release()
        self._bond_program.release()

    def _upload_box_buffer(self) -> None:
        if self._box_program is None or not self._box_vao.isCreated():
            return
        self._box_program.bind()
        self._box_vao.bind()

        self._box_vbo.bind()
        self._box_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
        if self._box_vertex_count > 0:
            self._box_vbo.allocate(self._box_vertices.tobytes(), int(self._box_vertices.nbytes))
        else:
            self._box_vbo.allocate(1)
        self._box_program.enableAttributeArray(7)
        self._box_program.setAttributeBuffer(7, GL_FLOAT, 0, 3, 0)

        self._box_vao.release()
        self._box_program.release()
        self._box_buffer_dirty = False

    def _upload_frame_buffers(self) -> None:
        if self._atom_count <= 0 or not self._position_vbos or not self._position_textures:
            return
        next_index = (self._position_buffer_index + 1) % len(self._position_vbos)
        buffer = self._position_vbos[next_index]
        if not buffer.isCreated():
            return
        upload_start = time.perf_counter()
        self._write_buffer(buffer, self._positions)
        self._last_upload_ms = (time.perf_counter() - upload_start) * 1000.0
        self._position_buffer_index = next_index
        self._position_vbo = buffer
        self._position_texture = self._position_textures[next_index]
        self._positions_uploaded = True
        self._frame_upload_pending = False

    def _write_buffer(self, buffer: QOpenGLBuffer, array: np.ndarray) -> None:
        buffer.bind()
        ptr = VoidPtr(int(array.ctypes.data), int(array.nbytes), False)
        try:
            buffer.write(0, ptr, int(array.nbytes))
        except (TypeError, ValueError):
            buffer.allocate(array.tobytes(), int(array.nbytes))

    def _radii_for_render_mode(self, mode: str) -> np.ndarray:
        return self._vdw_radii

    def _set_visible_atom_arrays(self, indices: np.ndarray) -> None:
        self._visible_atom_indices = np.ascontiguousarray(indices, dtype=np.int32)
        self._visible_atom_count = int(indices.shape[0])
        self._atom_visible_mask = np.zeros(self._atom_count, dtype=np.bool_)
        self._atom_visible_mask[indices] = True
        self._depth_order_dirty = True
        self._rebuild_render_atom_arrays()

    def _rebuild_render_atom_arrays(self) -> None:
        indices = self._visible_atom_indices
        render_indices = indices
        if (
            self._conservative_depth_enabled
            and indices.size > 1
            and self._positions.shape == (self._atom_count, 3)
        ):
            yaw = math.radians(self._yaw)
            pitch = math.radians(self._pitch)
            camera_forward = np.array(
                [
                    -math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch),
                    math.cos(pitch) * math.cos(yaw),
                ],
                dtype=np.float32,
            )
            view_depth = self._display_positions(indices) @ camera_forward
            order = np.argsort(view_depth, kind="stable")[::-1]
            render_indices = indices[order]
        self._render_atom_indices = np.ascontiguousarray(render_indices, dtype=np.int32)
        self._visible_atom_indices_gpu = np.ascontiguousarray(render_indices, dtype=np.float32)
        self._visible_unwrap_anchor_indices_gpu = np.ascontiguousarray(
            self._atom_unwrap_anchor_indices[render_indices],
            dtype=np.float32,
        )
        self._visible_radii = np.ascontiguousarray(self._radii[render_indices], dtype=np.float32)
        self._visible_colors = np.ascontiguousarray(self._colors[render_indices], dtype=np.float32)
        self._depth_order_dirty = False

    def _rebuild_visible_bonds(self) -> None:
        if self._bond_pairs.size == 0 or self._atom_visible_mask.size == 0:
            visible_pairs = np.empty((0, 2), dtype=np.int32)
        elif self._visible_atom_count == self._atom_count:
            visible_pairs = self._bond_pairs
        elif self._visible_atom_count <= 1:
            visible_pairs = np.empty((0, 2), dtype=np.int32)
        else:
            visible = self._atom_visible_mask[self._bond_pairs[:, 0]]
            visible &= self._atom_visible_mask[self._bond_pairs[:, 1]]
            visible_pairs = np.ascontiguousarray(self._bond_pairs[visible], dtype=np.int32)

        bond_count = int(visible_pairs.shape[0])
        self._bond_instance_count = bond_count
        self._bond_instance_data = np.empty((bond_count, 5), dtype=np.float32)
        self._bond_instance_colors = np.empty((bond_count, 6), dtype=np.float32)
        if bond_count == 0:
            return
        self._bond_instance_data[:, 0:2] = visible_pairs
        self._bond_instance_data[:, 2] = self._radii[visible_pairs[:, 0]]
        self._bond_instance_data[:, 3] = self._radii[visible_pairs[:, 1]]
        self._bond_instance_data[:, 4] = self._atom_unwrap_anchor_indices[
            visible_pairs[:, 0]
        ]
        self._bond_instance_colors = bond_segment_colors_for_pairs(
            self._colors,
            visible_pairs,
        ).reshape(bond_count, 6)

    def _display_positions(self, indices: np.ndarray) -> np.ndarray:
        anchor_indices = self._atom_unwrap_anchor_indices[indices]
        if (
            not np.any(anchor_indices >= 0)
            or not self._periodic_cell_valid
        ):
            if indices.size == self._atom_count:
                return self._positions
            return np.ascontiguousarray(self._positions[indices], dtype=np.float32)
        return unwrap_positions_by_anchor_indices(
            self._positions,
            indices,
            anchor_indices,
            self._box_cell,
            inverse_columns=self._cell_inverse_columns,
        )

    def _visible_positions(self) -> np.ndarray:
        return self._display_positions(self._visible_atom_indices)

    def _reset_bonds(self) -> None:
        self._bond_pairs = np.empty((0, 2), dtype=np.int32)
        self._bond_instance_count = 0
        self._bond_instance_data = np.empty((0, 5), dtype=np.float32)
        self._bond_instance_colors = np.empty((0, 6), dtype=np.float32)

    def _fit_camera_to_frame(
        self,
        positions: np.ndarray,
        *,
        include_box: bool = True,
        minimum_radius: float = 1.0,
    ) -> None:
        if positions.size == 0:
            self._center = QVector3D(0.0, 0.0, 0.0)
            self._scene_radius = 10.0
            self._distance = 40.0
            return
        points = positions
        if include_box and self._box_vertex_count > 0:
            points = np.concatenate((positions, self._box_vertices), axis=0)
        center = points.mean(axis=0, dtype=np.float64)
        span = points - center.astype(np.float32)
        radius = float(np.sqrt(np.max(np.sum(span * span, axis=1)))) if len(positions) else 10.0
        self._center = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self._scene_radius = max(radius, float(minimum_radius))
        self._distance = self._scene_radius * CAMERA_DISTANCE_SCALE

    def _camera_matrices(self) -> tuple[QMatrix4x4, QMatrix4x4]:
        aspect = max(1.0, self.width() / max(1, self.height()))
        proj = QMatrix4x4()
        proj.perspective(45.0, aspect, 0.01, max(1000.0, self._scene_radius * 100.0))

        view = QMatrix4x4()
        view.translate(0.0, 0.0, -self._distance)
        view.rotate(self._pitch, 1.0, 0.0, 0.0)
        view.rotate(self._yaw, 0.0, 1.0, 0.0)
        view.translate(-self._center)
        return view, proj
