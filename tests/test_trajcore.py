import unittest
from unittest.mock import patch

import numpy as np

from trajplayer import trajcore
from trajplayer.trajcore import (
    candidate_pairs,
    coarse_depth_order,
    connected_components,
    select_valence_bonds,
)


class TrajcoreTests(unittest.TestCase):
    @unittest.skipUnless(
        trajcore.NATIVE_POSITION_DEPTH_ORDER_AVAILABLE,
        "native projected depth ordering is not built",
    )
    def test_native_projected_depth_order_matches_existing_visual_bins(self) -> None:
        x = np.linspace(-4.0, 7.0, 100_000, dtype=np.float32)
        positions = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        actual = trajcore.coarse_position_depth_order(positions, forward)
        expected = coarse_depth_order(positions @ forward)

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(
        trajcore.NATIVE_XYZ_READ_AVAILABLE,
        "native trajcore XYZ parser is not built",
    )
    def test_native_xyz_parser_writes_reordered_columns_into_caller_buffer(self) -> None:
        rows = (
            b"discard C 1.25 -2e-1 .5 9\n"
            b"discard O -4.0 5.5 6.25 8\n"
        )
        positions = np.full((2, 3), np.nan, dtype=np.float32)

        used_native = trajcore.xyz_read_frame_into(
            rows,
            data_offset=0,
            data_end=len(rows),
            positions=positions,
            identity_column=1,
            identity_is_atomic_number=False,
            position_columns=(2, 3, 4),
            expected_columns=6,
            expected_atom_numbers=np.array([6, 8], dtype=np.uint16),
        )

        self.assertTrue(used_native)
        np.testing.assert_allclose(
            positions,
            [[1.25, -0.2, 0.5], [-4.0, 5.5, 6.25]],
        )

    @unittest.skipUnless(
        trajcore.NATIVE_XYZ_READ_AVAILABLE,
        "native trajcore XYZ parser is not built",
    )
    def test_native_xyz_parser_validates_atomic_order(self) -> None:
        rows = b"1.0 2.0 3.0 8\n"
        positions = np.empty((1, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "identity differs"):
            trajcore.xyz_read_frame_into(
                rows,
                data_offset=0,
                data_end=len(rows),
                positions=positions,
                identity_column=3,
                identity_is_atomic_number=True,
                position_columns=(0, 1, 2),
                expected_columns=4,
                expected_atom_numbers=np.array([6], dtype=np.uint16),
            )

    def test_coarse_depth_order_matches_previous_stable_bin_order(self) -> None:
        rng = np.random.default_rng(20260810)
        depth = rng.normal(size=50_000).astype(np.float32)
        depth[::127] = np.float32(0.25)
        minimum = float(np.min(depth))
        maximum = float(np.max(depth))
        bins = np.asarray(
            np.clip(
                (depth - minimum) * ((trajcore.DEPTH_BIN_COUNT - 1) / (maximum - minimum)),
                0,
                trajcore.DEPTH_BIN_COUNT - 1,
            ),
            dtype=np.uint8,
        )
        expected = np.argsort(bins, kind="stable")[::-1]

        np.testing.assert_array_equal(coarse_depth_order(depth), expected)

    def test_coarse_depth_order_handles_degenerate_and_nonfinite_input(self) -> None:
        np.testing.assert_array_equal(
            coarse_depth_order(np.ones(5, dtype=np.float32)),
            np.array([4, 3, 2, 1, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            coarse_depth_order(np.array([0.0, np.nan, 1.0], dtype=np.float32)),
            np.array([2, 1, 0], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            coarse_depth_order(np.zeros((2, 2), dtype=np.float32))

    def test_python_depth_fallback_matches_public_order(self) -> None:
        depth = np.linspace(-2.0, 3.0, 1024, dtype=np.float32)
        with patch("trajplayer.trajcore._native", None):
            fallback = coarse_depth_order(depth)
        np.testing.assert_array_equal(fallback, trajcore._python_coarse_depth_order(depth))

    def test_connected_components_returns_compact_stable_labels(self) -> None:
        labels, sizes = connected_components(
            7,
            np.array([[0, 1], [2, 3], [1, 2], [5, 6]], dtype=np.int32),
        )
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0, 1, 2, 2]))
        np.testing.assert_array_equal(sizes, np.array([4, 1, 2]))

    def test_valence_selection_matches_lexicographic_greedy_reference(self) -> None:
        rng = np.random.default_rng(20260810)
        atom_count = 250
        candidate_count = 20_000
        left = rng.integers(0, atom_count - 1, size=candidate_count, dtype=np.int32)
        right = rng.integers(1, atom_count, size=candidate_count, dtype=np.int32)
        np.minimum(left, right, out=left)
        np.maximum(left + 1, right, out=right)
        np.minimum(right, atom_count - 1, out=right)
        distance2 = rng.integers(1, 100, size=candidate_count).astype(np.float32)
        caps = rng.integers(0, 7, size=atom_count, dtype=np.uint8)
        expected = trajcore._python_select_valence_bonds(distance2, left, right, caps)

        actual = select_valence_bonds(distance2, left, right, caps)

        np.testing.assert_array_equal(actual, expected)

    def test_valence_selection_validates_candidate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "indices must be in range"):
            select_valence_bonds(
                np.array([1.0], dtype=np.float32),
                np.array([0], dtype=np.int32),
                np.array([2], dtype=np.int32),
                np.ones(2, dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            select_valence_bonds(
                np.array([1.0], dtype=np.float32),
                np.array([1], dtype=np.int32),
                np.array([1], dtype=np.int32),
                np.ones(2, dtype=np.uint8),
            )

    @unittest.skipUnless(
        trajcore.NATIVE_VALENCE_SELECTION_AVAILABLE,
        "native valence selection is not built",
    )
    def test_native_valence_selection_matches_python_reference(self) -> None:
        distance2 = np.array([0.5, 0.2, 0.2, 0.2, 0.8], dtype=np.float32)
        left = np.array([0, 0, 0, 1, 2], dtype=np.int32)
        right = np.array([1, 2, 3, 2, 3], dtype=np.int32)
        caps = np.array([2, 1, 1, 1], dtype=np.uint8)

        expected = trajcore._python_select_valence_bonds(distance2, left, right, caps)
        actual = select_valence_bonds(distance2, left, right, caps)

        np.testing.assert_array_equal(actual, expected)

    def test_nonperiodic_cell_list_matches_brute_force(self) -> None:
        rng = np.random.default_rng(12)
        positions = rng.uniform(-5.0, 5.0, size=(120, 3)).astype(np.float32)
        active = np.arange(positions.shape[0], dtype=np.int32)
        self._assert_matches_brute_force(positions, active, cutoff=1.75, cell=None)

    def test_triclinic_periodic_cell_list_matches_brute_force(self) -> None:
        rng = np.random.default_rng(22)
        cell = np.array(
            [[9.0, 0.0, 0.0], [1.5, 8.0, 0.0], [0.5, 1.0, 7.0]],
            dtype=np.float64,
        )
        fractional = rng.uniform(0.0, 1.0, size=(100, 3))
        positions = np.asarray(fractional @ cell, dtype=np.float32)
        active = np.arange(positions.shape[0], dtype=np.int32)
        self._assert_matches_brute_force(positions, active, cutoff=1.6, cell=cell)

    def test_python_periodic_fallback_has_no_ase_or_scipy_dependency(self) -> None:
        positions = np.array([[0.2, 0, 0], [9.4, 0, 0]], dtype=np.float32)
        active = np.arange(2, dtype=np.int32)
        cell = np.diag([10.0, 10.0, 10.0])
        with patch("trajplayer.trajcore._native", None):
            distance2, left, right = candidate_pairs(
                positions,
                active,
                1.0,
                cell=cell,
            )
        np.testing.assert_allclose(distance2, [0.64], atol=1.0e-5)
        np.testing.assert_array_equal(left, [0])
        np.testing.assert_array_equal(right, [1])

    def _assert_matches_brute_force(
        self,
        positions: np.ndarray,
        active: np.ndarray,
        *,
        cutoff: float,
        cell: np.ndarray | None,
    ) -> None:
        distance2, left, right = candidate_pairs(
            positions,
            active,
            cutoff,
            cell=cell,
        )
        actual = {
            (int(i), int(j)): float(value)
            for value, i, j in zip(distance2, left, right)
        }
        expected: dict[tuple[int, int], float] = {}
        inverse = None if cell is None else np.linalg.inv(cell)
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                delta = np.asarray(positions[j] - positions[i], dtype=np.float64)
                if cell is not None and inverse is not None:
                    fractional = delta @ inverse
                    fractional -= np.rint(fractional)
                    delta = fractional @ cell
                value = float(np.dot(delta, delta))
                if value <= cutoff * cutoff:
                    expected[(i, j)] = value
        self.assertEqual(set(actual), set(expected))
        for pair, value in expected.items():
            self.assertAlmostEqual(actual[pair], value, places=4)


if __name__ == "__main__":
    unittest.main()
