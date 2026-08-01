import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from trajplayer.ase_cache import build_cache_from_ase, inspect_ase_source, open_valid_cache


class AseCacheTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows file locking regression")
    def test_rebuild_while_cache_is_mapped_uses_temporary_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "locked.traj"
            with Trajectory(str(source), "w") as traj:
                traj.write(Atoms("H", positions=[[0.0, 0.0, 0.0]]))

            first = build_cache_from_ase(source)
            canonical_root = first.root
            second = build_cache_from_ase(source)
            temporary_root = second.root
            try:
                self.assertEqual(canonical_root, source.with_name(f"{source.name}.tpdata"))
                self.assertNotEqual(temporary_root, canonical_root)
                self.assertTrue(second.metadata["temporary_cache"])
                self.assertTrue(canonical_root.exists())
            finally:
                second.close()
                first.close()
            self.assertFalse(temporary_root.exists())

    def test_build_cache_from_ase_traj_streams_to_float32_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tiny.traj"
            with Trajectory(str(source), "w") as traj:
                for frame_index in range(4):
                    atoms = Atoms(
                        "HCO",
                        positions=np.array(
                            [
                                [frame_index, 0.0, 0.0],
                                [0.0, frame_index + 1.0, 0.0],
                                [0.0, 0.0, frame_index + 2.0],
                            ],
                            dtype=np.float64,
                        ),
                        cell=np.diag([10.0 + frame_index, 11.0, 12.0]),
                        pbc=True,
                    )
                    traj.write(atoms)

            summary = inspect_ase_source(source)
            self.assertEqual(summary.frame_count, 4)
            self.assertEqual(summary.atom_count, 3)
            self.assertEqual(summary.symbols, ["H", "C", "O"])

            progress: list[tuple[int, int]] = []
            previews: list[int] = []

            def capture_preview(store) -> None:
                self.assertFalse(store.is_complete)
                self.assertEqual(store.available_frame_count, 1)
                np.testing.assert_array_equal(store.frame(0), np.array([[0, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float32))
                previews.append(store.available_frame_count)

            with build_cache_from_ase(
                source,
                progress_callback=lambda done, total: progress.append((done, total)),
                preview_callback=capture_preview,
            ) as store:
                self.assertEqual(store.positions.dtype, np.float32)
                self.assertEqual(store.positions.shape, (4, 3, 3))
                self.assertTrue(store.has_cells)
                np.testing.assert_array_equal(
                    store.frame(2),
                    np.array(
                        [
                            [2.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0],
                            [0.0, 0.0, 4.0],
                        ],
                        dtype=np.float32,
                    ),
                )
                np.testing.assert_array_equal(
                    store.cell(2),
                    np.diag([12.0, 11.0, 12.0]).astype(np.float32),
                )
                self.assertTrue(store.is_valid_for_source(source))
                self.assertTrue(store.is_complete)
                self.assertEqual(store.available_frame_count, 4)

            self.assertEqual(previews, [1])
            self.assertEqual(progress[-1], (4, 4))
            with open_valid_cache(source) as reopened:
                self.assertEqual(reopened.frame_count, 4)
                np.testing.assert_array_equal(reopened.atom_numbers, np.array([1, 6, 8], dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
