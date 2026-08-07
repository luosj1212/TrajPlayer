import unittest

import numpy as np

from trajplayer.bonds import (
    bond_segments_for_frame,
    connected_components,
    infer_bonds,
    valence_caps_for,
)


class BondInferenceTests(unittest.TestCase):
    def test_water_infers_two_oh_bonds_without_hh_bond(self) -> None:
        positions = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.958, 0.000, 0.000],
                [-0.240, 0.927, 0.000],
            ],
            dtype=np.float32,
        )
        atom_numbers = np.array([8, 1, 1], dtype=np.uint16)

        bonds = infer_bonds(positions, atom_numbers)

        self.assertEqual(bonds.dtype, np.int32)
        self.assertTrue(bonds.flags["C_CONTIGUOUS"])
        self.assertEqual({tuple(pair) for pair in bonds.tolist()}, {(0, 1), (0, 2)})

    def test_valence_caps_prevent_overbonding(self) -> None:
        positions = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.95, 0.00, 0.00],
                [0.97, 0.00, 0.00],
                [0.99, 0.00, 0.00],
            ],
            dtype=np.float32,
        )
        atom_numbers = np.array([8, 1, 1, 1], dtype=np.uint16)

        bonds = infer_bonds(positions, atom_numbers)

        self.assertEqual({tuple(pair) for pair in bonds.tolist()}, {(0, 1), (0, 2)})
        np.testing.assert_array_equal(valence_caps_for(atom_numbers), np.array([2, 1, 1, 1], dtype=np.uint8))

    def test_periodic_inference_finds_bond_across_cell_boundary(self) -> None:
        positions = np.array(
            [
                [0.2, 0.0, 0.0],
                [9.3, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        atom_numbers = np.array([6, 1], dtype=np.uint16)
        cell = np.diag([10.0, 10.0, 10.0]).astype(np.float32)

        nonperiodic_bonds = infer_bonds(positions, atom_numbers)
        periodic_bonds = infer_bonds(positions, atom_numbers, cell=cell)

        self.assertEqual(nonperiodic_bonds.shape, (0, 2))
        np.testing.assert_array_equal(periodic_bonds, np.array([[0, 1]], dtype=np.int32))

    def test_bond_segments_are_split_at_midpoint_and_colored_by_endpoint_atoms(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        bonds = np.array([[0, 1]], dtype=np.int32)
        colors = np.array([[0.2, 0.2, 0.2], [1.0, 0.0, 0.0]], dtype=np.float32)

        starts, ends, segment_colors = bond_segments_for_frame(positions, bonds, colors)

        self.assertEqual(starts.shape, (2, 3))
        self.assertTrue(starts.flags["C_CONTIGUOUS"])
        self.assertTrue(ends.flags["C_CONTIGUOUS"])
        self.assertTrue(segment_colors.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(starts, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_allclose(ends, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(segment_colors, colors)

    def test_connected_components_label_selectable_chains(self) -> None:
        bonds = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32)

        component_ids, component_sizes = connected_components(6, bonds)

        np.testing.assert_array_equal(component_ids, np.array([0, 0, 0, 1, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(component_sizes, np.array([3, 2, 1], dtype=np.int32))
        self.assertTrue(component_ids.flags.c_contiguous)
        self.assertTrue(component_sizes.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
