import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from trajplayer.binary_store import (
    BinaryTrajectoryStore,
    CacheValidationError,
    MAX_METADATA_BYTES,
    METADATA_FILE,
    POSITIONS_FILE,
    cache_dir_for_source,
    prepare_cache_directory,
)


class BinaryTrajectoryStoreTests(unittest.TestCase):
    def test_create_rejects_out_of_range_atomic_numbers_before_narrowing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "values from 0 to 118"):
                BinaryTrajectoryStore.create(
                    Path(tmp) / "bad-number.tpdata",
                    frame_count=1,
                    atom_numbers=np.array([70000], dtype=np.int64),
                    symbols=None,
                    source_path=None,
                    source_mtime_ns=0,
                    source_size=0,
                )

    def test_prepare_rejects_linked_cache_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "important"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            root = Path(tmp) / "sample.traj.tpdata"
            try:
                root.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(CacheValidationError, "linked trajectory cache"):
                prepare_cache_directory(root)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_prepare_rejects_windows_reparse_directory_without_removing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "junction.tpdata"
            root.mkdir()
            marker = root / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with patch(
                "trajplayer.binary_store._is_link_or_reparse_point",
                return_value=True,
            ):
                with self.assertRaisesRegex(CacheValidationError, "linked trajectory cache"):
                    prepare_cache_directory(root)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    @unittest.skipUnless(os.name == "nt", "Windows file locking regression")
    def test_locked_cache_directory_uses_temporary_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "locked.tpdata"
            root.mkdir()
            mapped = np.memmap(root / "atom_numbers.u16", dtype=np.uint16, mode="w+", shape=(2,))
            try:
                prepared, temporary = prepare_cache_directory(root)
                self.assertTrue(temporary)
                self.assertNotEqual(prepared, root)
                self.assertTrue(root.exists())
                self.assertTrue(prepared.exists())
                prepared.rmdir()
            finally:
                mapped._mmap.close()

    def test_temporary_store_removes_its_cache_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session.tpdata"
            store = BinaryTrajectoryStore.create(
                root,
                frame_count=1,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
                temporary_cache=True,
            )
            self.assertTrue(root.exists())
            store.close()
            self.assertFalse(root.exists())

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

    def test_source_identity_rejects_same_size_same_mtime_content_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.traj"
            source.write_bytes(b"abc")
            source_stat = source.stat()
            root = cache_dir_for_source(source)
            with BinaryTrajectoryStore.create(
                root,
                frame_count=1,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=source,
                source_mtime_ns=source_stat.st_mtime_ns,
                source_size=source_stat.st_size,
            ):
                pass

            source.write_bytes(b"xyz")
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            self.assertEqual(source.stat().st_size, source_stat.st_size)
            self.assertEqual(source.stat().st_mtime_ns, source_stat.st_mtime_ns)

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

    def test_open_rejects_truncated_position_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "truncated.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=2,
                atom_numbers=np.array([6, 1], dtype=np.uint16),
                symbols=["C", "H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ):
                pass
            with (root / POSITIONS_FILE).open("r+b") as handle:
                handle.truncate(12)

            with self.assertRaisesRegex(CacheValidationError, "positions.f32 size"):
                BinaryTrajectoryStore.open(root)

    def test_open_rejects_availability_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traversal.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=2,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
                progressive=True,
            ):
                pass
            metadata_path = root / METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frame_availability_file"] = "../outside.u8"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(CacheValidationError, "availability member"):
                BinaryTrajectoryStore.open(root)

    def test_open_rejects_cell_shape_that_does_not_match_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bad-cell-shape.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=2,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
                store_cells=True,
            ):
                pass
            metadata_path = root / METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["cell_shape"] = [1, 3, 3]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(CacheValidationError, "Invalid cell shape"):
                BinaryTrajectoryStore.open(root)

    def test_open_rejects_dimensions_that_overflow_platform_address_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "oversized.tpdata"
            with BinaryTrajectoryStore.create(
                root,
                frame_count=1,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ):
                pass
            metadata_path = root / METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frame_count"] = sys.maxsize
            metadata["shape"] = [sys.maxsize, 1, 3]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(CacheValidationError, "dimensions are too large"):
                BinaryTrajectoryStore.open(root)

    def test_open_rejects_oversized_or_malformed_metadata_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            oversized_root = Path(tmp) / "oversized-metadata.tpdata"
            oversized_root.mkdir()
            (oversized_root / METADATA_FILE).write_bytes(b" " * (MAX_METADATA_BYTES + 1))
            with self.assertRaisesRegex(CacheValidationError, "metadata is too large"):
                BinaryTrajectoryStore.open(oversized_root)

            malformed_root = Path(tmp) / "malformed-metadata.tpdata"
            with BinaryTrajectoryStore.create(
                malformed_root,
                frame_count=1,
                atom_numbers=np.array([6], dtype=np.uint16),
                symbols=["C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ):
                pass
            metadata_path = malformed_root / METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source"] = ["not", "an", "object"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(CacheValidationError, "'source' must be an object"):
                BinaryTrajectoryStore.open(malformed_root)


if __name__ == "__main__":
    unittest.main()
