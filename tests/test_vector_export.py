from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest

from trajplayer.vector_export import VectorSceneSnapshot, write_molecule_svg


SVG = "{http://www.w3.org/2000/svg}"


def perspective_matrix(
    *,
    field_of_view_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    scale = 1.0 / math.tan(math.radians(field_of_view_degrees) * 0.5)
    return np.array(
        [
            [scale / aspect, 0.0, 0.0, 0.0],
            [0.0, scale, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def scene() -> VectorSceneSnapshot:
    view = np.eye(4, dtype=np.float64)
    view[2, 3] = -10.0
    return VectorSceneSnapshot(
        width=800,
        height=600,
        view_matrix=view,
        projection_matrix=perspective_matrix(
            field_of_view_degrees=45.0,
            aspect=800.0 / 600.0,
            near=0.01,
            far=1000.0,
        ),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        atom_indices=np.array([0, 1], dtype=np.int32),
        atom_radii=np.array([1.7, 1.52], dtype=np.float32),
        atom_colors=np.array([[0.56, 0.56, 0.56], [1.0, 0.05, 0.05]], dtype=np.float32),
        atom_selected=np.array([0, 1], dtype=np.uint8),
        selection_color=(0.02, 0.58, 0.74),
        bond_pairs=np.array([[0, 1]], dtype=np.int32),
        bond_colors=np.array([[[0.55, 0.55, 0.54], [0.96, 0.07, 0.07]]], dtype=np.float32),
        atom_size_scale=0.25,
        bond_size_scale=1.0,
        bond_radius=0.20,
        bond_endpoint_radius_scale=0.92,
        show_atoms=True,
        show_bonds=True,
        periodic_cell=None,
        box_segments=np.empty((0, 2, 3), dtype=np.float32),
        box_color=(0.18, 0.18, 0.18),
        background=(1.0, 1.0, 1.0),
    )


def test_svg_is_true_vector_current_viewport_with_atoms_and_two_tone_bond(tmp_path) -> None:
    path = tmp_path / "molecule.svg"
    write_molecule_svg(path, scene())

    root = ET.parse(path).getroot()
    assert root.tag == f"{SVG}svg"
    assert root.attrib["viewBox"] == "0 0 800 600"
    assert root.find(f".//{SVG}image") is None
    atoms = root.findall(f".//{SVG}ellipse[@class='atom']")
    bonds = root.findall(f".//{SVG}g[@class='bond']")
    assert len(atoms) == 2
    assert len(bonds) == 1
    assert len(bonds[0].findall(f"{SVG}rect")) == 2
    assert float(atoms[0].attrib["cx"]) == pytest.approx(400.0, abs=0.01)
    assert float(atoms[1].attrib["cx"]) > float(atoms[0].attrib["cx"])
    assert all(float(atom.attrib["rx"]) > 0.0 for atom in atoms)
    source = path.read_text(encoding="utf-8")
    assert "data-atom-index=\"0\"" in source
    assert "atom-" in source
    assert "bond-" in source


def test_svg_respects_current_render_visibility_and_periodic_box(tmp_path) -> None:
    original = scene()
    vector_scene = replace(
        original,
        show_atoms=False,
        box_segments=np.array(
            [[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=np.float32,
        ),
    )
    path = tmp_path / "bond-only.svg"
    write_molecule_svg(path, vector_scene)

    root = ET.parse(path).getroot()
    assert root.findall(f".//{SVG}ellipse[@class='atom']") == []
    assert len(root.findall(f".//{SVG}g[@class='bond']")) == 1
    assert len(root.findall(f".//{SVG}line[@class='periodic-box']")) == 1


def test_svg_export_honors_cancellation_before_writing(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="cancelled"):
        write_molecule_svg(
            tmp_path / "cancelled.svg",
            scene(),
            cancelled=lambda: True,
        )
