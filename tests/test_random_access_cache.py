import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory

from trajplayer.random_access_cache import (
    FRAME_OFFSETS_FILE,
    open_direct_random_access_store,
    open_random_access_session,
    write_reader_frame,
)
from trajplayer.trajectory_source import TrajectorySource
from trajplayer.xyz_index import ProgressiveXyzIndex


class RandomAccessCacheTests(unittest.TestCase):
    def test_extxyz_direct_store_exposes_first_frame_before_index_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "progressive.extxyz"
            frames = [
                Atoms("H", positions=[[float(frame_index), 0.0, 0.0]])
                for frame_index in range(8)
            ]
            write(source, frames, format="extxyz")

            with patch.object(ProgressiveXyzIndex, "start", return_value=None):
                store = open_direct_random_access_store(TrajectorySource(source))
            try:
                self.assertEqual(store.frame_count, 1)
                self.assertFalse(store.frame_count_is_final)
                self.assertTrue(store.metadata["direct_reader"])
                self.assertFalse((store.root / "positions.f32").exists())
                positions, _cell = store.read_frame_arrays(0)
                np.testing.assert_allclose(positions, frames[0].positions)
            finally:
                store.close()

    def test_extxyz_progressive_index_publishes_final_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "progressive.extxyz"
            frames = [
                Atoms("HC", positions=[[frame_index, 0.0, 0.0], [0.0, frame_index, 0.0]])
                for frame_index in range(25)
            ]
            write(source, frames, format="extxyz")
            store = open_direct_random_access_store(TrajectorySource(source))
            try:
                count = store.frame_count
                while not store.frame_count_is_final:
                    count, _complete = store.wait_for_index_update(count, timeout_s=1.0)
                self.assertEqual(store.frame_count, len(frames))
                positions, _cell = store.read_frame_arrays(len(frames) - 1)
                np.testing.assert_allclose(positions, frames[-1].positions)
            finally:
                store.close()

    def test_ase_traj_writes_requested_frame_before_sequential_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "random.traj"
            with Trajectory(str(source), "w") as trajectory:
                for frame_index in range(5):
                    trajectory.write(
                        Atoms(
                            "HC",
                            positions=np.array(
                                [[frame_index, 0.0, 0.0], [0.0, frame_index + 1.0, 0.0]],
                                dtype=np.float64,
                            ),
                        )
                    )

            reader, store = open_random_access_session(TrajectorySource(source))
            try:
                write_reader_frame(reader, store, 4)
                self.assertTrue(store.is_frame_available(4))
                self.assertFalse(store.is_frame_available(0))
                np.testing.assert_array_equal(
                    store.frame(4),
                    np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32),
                )
            finally:
                reader.close()
                store.close()

            reader, store = open_random_access_session(TrajectorySource(source))
            try:
                self.assertEqual(store.available_frame_count, 1)
                self.assertTrue(store.is_frame_available(4))
            finally:
                reader.close()
                store.close()

    def test_extxyz_builds_disk_index_and_reads_frames_by_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "indexed.extxyz"
            frames = []
            for frame_index in range(4):
                frames.append(
                    Atoms(
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
                )
            write(source, frames, format="extxyz")

            reader, store = open_random_access_session(TrajectorySource(source))
            try:
                self.assertEqual(store.frame_count, 4)
                self.assertTrue((store.root / FRAME_OFFSETS_FILE).exists())
                write_reader_frame(reader, store, 3)
                self.assertTrue(store.is_frame_available(3))
                self.assertFalse(store.is_frame_available(0))
                np.testing.assert_allclose(store.frame(3), frames[3].positions)
                np.testing.assert_allclose(store.cell(3), frames[3].cell.array)
            finally:
                reader.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
