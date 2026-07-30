import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajplayer.binary_store import BinaryTrajectoryStore, cache_dir_for_source


class BinaryTrajectoryStoreTests(unittest.TestCase):
    def test_store_reopens_positions_as_contiguous_float32_memmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sample.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=3,
                atom_numbers=np.array([6, 1], dtype=np.uint16),
                symbols=["C", "H"],
                source_path=Path(tmp) / "sample.traj",
                source_mtime_ns=123,
                source_size=456,
            ) as store:
                store.positions[:] = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
                store.flush()

            with BinaryTrajectoryStore.open(root) as reopened:
                self.assertEqual(reopened.frame_count, 3)
                self.assertEqual(reopened.atom_count, 2)
                self.assertEqual(reopened.positions.shape, (3, 2, 3))
                self.assertEqual(reopened.positions.dtype, np.float32)
                self.assertIsInstance(reopened.positions, np.memmap)
                self.assertTrue(reopened.frame(1).flags["C_CONTIGUOUS"])
                np.testing.assert_array_equal(
                    reopened.frame(1),
                    np.array([[6, 7, 8], [9, 10, 11]], dtype=np.float32),
                )

    def test_store_reopens_cells_as_optional_float32_memmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sample.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=2,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=Path(tmp) / "sample.traj",
                source_mtime_ns=123,
                source_size=456,
                store_cells=True,
            ) as store:
                self.assertTrue(store.has_cells)
                store.cells[:] = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
                store.flush()

            with BinaryTrajectoryStore.open(root) as reopened:
                self.assertTrue(reopened.has_cells)
                self.assertIsInstance(reopened.cells, np.memmap)
                np.testing.assert_array_equal(
                    reopened.cell(1),
                    np.arange(9, 18, dtype=np.float32).reshape(3, 3),
                )

    def test_source_identity_rejects_stale_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.traj"
            source.write_bytes(b"abc")
            root = cache_dir_for_source(source)
            with BinaryTrajectoryStore.create(
                root,
                frame_count=1,
                atom_numbers=np.array([8], dtype=np.uint16),
                symbols=["O"],
                source_path=source,
                source_mtime_ns=source.stat().st_mtime_ns,
                source_size=source.stat().st_size,
            ) as store:
                store.flush()

            with BinaryTrajectoryStore.open(root) as reopened:
                self.assertTrue(reopened.is_valid_for_source(source))
            source.write_bytes(b"abcd")
            with BinaryTrajectoryStore.open(root) as reopened:
                self.assertFalse(reopened.is_valid_for_source(source))

    def test_progressive_store_only_publishes_completed_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "progressive.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=3,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
                progressive=True,
            ) as store:
                self.assertFalse(store.is_complete)
                self.assertEqual(store.available_frame_count, 0)
                self.assertFalse(store.is_frame_available(0))

                store.positions[0] = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
                store.mark_frame_available(1)
                self.assertTrue(store.is_frame_available(0))
                self.assertFalse(store.is_frame_available(1))
                with self.assertRaises(RuntimeError):
                    store.mark_complete()

                store.mark_frame_available(3)
                store.mark_complete()
                self.assertTrue(store.is_complete)

            with BinaryTrajectoryStore.open(root) as reopened:
                self.assertTrue(reopened.is_complete)
                self.assertEqual(reopened.available_frame_count, 3)

    def test_random_access_store_persists_out_of_order_frame_availability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "random.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=5,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
                progressive=True,
                random_access=True,
            ) as store:
                store.positions[4] = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
                store.publish_frame(4)

                self.assertTrue(store.supports_random_access)
                self.assertEqual(store.available_frame_count, 1)
                self.assertEqual(store.available_prefix_count, 0)
                self.assertEqual(store.navigable_frame_count, 5)
                self.assertFalse(store.is_frame_available(0))
                self.assertTrue(store.is_frame_available(4))

            with BinaryTrajectoryStore.open(root) as reopened:
                self.assertEqual(reopened.available_frame_count, 1)
                self.assertFalse(reopened.is_frame_available(0))
                self.assertTrue(reopened.is_frame_available(4))
                np.testing.assert_array_equal(
                    reopened.frame(4),
                    np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
