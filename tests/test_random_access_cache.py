import json
import tempfile
import threading
import time
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
from trajplayer.xyz_index import IndexIoCoordinator, ProgressiveXyzIndex


class RandomAccessCacheTests(unittest.TestCase):
    def test_xyz_index_resumes_from_an_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "resume.extxyz"
            frames = [
                Atoms("HC", positions=[[index, 0.0, 0.0], [0.0, index, 0.0]])
                for index in range(500)
            ]
            write(source, frames, format="extxyz")
            cache_root = root / "resume.tpindex"
            coordinator = IndexIoCoordinator(
                resume_grace_s=0.0,
                background_throttle_s=0.002,
            )

            with (
                patch("trajplayer.xyz_index.INDEX_CHUNK_BYTES", 128),
                patch("trajplayer.xyz_index.INDEX_CHECKPOINT_BYTES", 256),
            ):
                index = ProgressiveXyzIndex(
                    source,
                    cache_root,
                    atom_count=2,
                    io_coordinator=coordinator,
                )
                index.start()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and index.scan_offset < 256:
                    time.sleep(0.005)
                self.assertGreaterEqual(index.scan_offset, 256)
                self.assertFalse(index.complete)
                index.close()

            checkpoint = json.loads(
                (cache_root / "frame_index.json").read_text(encoding="utf-8")
            )
            resume_offset = int(checkpoint["scan_offset"])
            self.assertGreater(resume_offset, 0)
            self.assertFalse(checkpoint["complete"])

            resumed = ProgressiveXyzIndex(source, cache_root, atom_count=2)
            try:
                self.assertEqual(resumed.scan_offset, resume_offset)
                resumed.start()
                self.assertEqual(resumed.wait_until_complete(), len(frames))
                self.assertTrue(resumed.complete)
                self.assertEqual(resumed.offset(len(frames) - 1) > 0, True)
            finally:
                resumed.close()

    def test_xyz_index_qos_pauses_until_foreground_grace_period_ends(self) -> None:
        coordinator = IndexIoCoordinator(
            resume_grace_s=0.04,
            background_throttle_s=0.0,
        )
        stop_event = threading.Event()
        background_ready = threading.Event()
        background_ready_times: list[float] = []

        def wait_for_background() -> None:
            if coordinator.wait_for_background_turn(stop_event):
                background_ready_times.append(time.monotonic())
                background_ready.set()

        with coordinator.foreground():
            thread = threading.Thread(target=wait_for_background)
            thread.start()
            pause_deadline = time.monotonic() + 0.5
            while coordinator.stats_snapshot().pause_count == 0:
                self.assertLess(time.monotonic(), pause_deadline)
                time.sleep(0.001)
            self.assertFalse(background_ready.is_set())
            foreground_release_started_s = time.monotonic()

        self.assertTrue(background_ready.wait(timeout=0.5))
        thread.join(timeout=1.0)
        self.assertGreaterEqual(
            background_ready_times[0] - foreground_release_started_s,
            0.03,
        )
        stats = coordinator.stats_snapshot()
        self.assertEqual(stats.foreground_reads, 1)
        self.assertGreaterEqual(stats.pause_count, 1)
        self.assertGreater(stats.paused_seconds, 0.04)

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

    def test_extxyz_first_display_reuses_initial_metadata_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "first-frame.extxyz"
            write(
                source,
                [Atoms("HC", positions=[[0, 0, 0], [1, 0, 0]])],
                format="extxyz",
            )

            from trajplayer import random_access_cache

            with patch.object(
                random_access_cache,
                "read_xyz_frame",
                wraps=random_access_cache.read_xyz_frame,
            ) as decode:
                store = open_direct_random_access_store(TrajectorySource(source))
                try:
                    positions = np.empty((2, 3), dtype=np.float32)
                    store.read_frame_into(0, positions, None)
                    self.assertEqual(decode.call_count, 1)
                    np.testing.assert_allclose(positions, [[0, 0, 0], [1, 0, 0]])
                finally:
                    store.close()

    def test_extxyz_index_construction_failure_releases_source_handles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "failed-index.extxyz"
            write(source, [Atoms("H", positions=[[0, 0, 0]])], format="extxyz")

            with patch(
                "trajplayer.random_access_cache.ProgressiveXyzIndex",
                side_effect=RuntimeError("index failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "index failed"):
                    open_direct_random_access_store(TrajectorySource(source))

            renamed = source.with_name("released.extxyz")
            source.replace(renamed)
            self.assertTrue(renamed.is_file())

    def test_extxyz_native_reader_writes_later_frame_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "native.extxyz"
            frames = [
                Atoms("HC", positions=[[index, 0, 0], [0, index + 1, 0]])
                for index in range(2)
            ]
            write(source, frames, format="extxyz")

            def fake_native(_source, **kwargs):
                np.copyto(
                    kwargs["positions"],
                    np.asarray(frames[1].positions, dtype=np.float32),
                )
                return True

            with (
                patch("trajplayer.random_access_cache.NATIVE_XYZ_READ_AVAILABLE", True),
                patch(
                    "trajplayer.random_access_cache.xyz_read_frame_into",
                    side_effect=fake_native,
                ) as native_read,
            ):
                store = open_direct_random_access_store(TrajectorySource(source))
                try:
                    count = store.frame_count
                    while not store.frame_count_is_final:
                        count, _complete = store.wait_for_index_update(count, timeout_s=1.0)
                    positions = np.empty((2, 3), dtype=np.float32)
                    store.read_frame_into(1, positions, None)
                    native_read.assert_called_once()
                    np.testing.assert_allclose(positions, frames[1].positions)
                    self.assertEqual(store.reader.native_read_count, 1)
                    self.assertEqual(store.reader.python_read_count, 0)
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
