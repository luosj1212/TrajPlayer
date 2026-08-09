import unittest
from unittest.mock import patch

import numpy as np

from trajplayer.trajcore import candidate_pairs, connected_components


class TrajcoreTests(unittest.TestCase):
    def test_connected_components_returns_compact_stable_labels(self) -> None:
        labels, sizes = connected_components(
            7,
            np.array([[0, 1], [2, 3], [1, 2], [5, 6]], dtype=np.int32),
        )
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0, 1, 2, 2]))
        np.testing.assert_array_equal(sizes, np.array([4, 1, 2]))

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
