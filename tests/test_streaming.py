import threading
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from trajplayer.binary_store import BinaryTrajectoryStore
from trajplayer.streaming import FrameStreamer


class FrameStreamerTests(unittest.TestCase):
    def test_memory_budget_caps_large_prefetch_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "budget.tpdata",
                frame_count=20,
                atom_numbers=np.array([1, 6], dtype=np.uint16),
                symbols=["H", "C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                frame_bytes = 2 * 3 * 4
                streamer = FrameStreamer(
                    store,
                    prefetch_radius=200,
                    max_memory_bytes=frame_bytes * 3,
                )

                self.assertEqual(streamer.capacity, 3)
                self.assertEqual(streamer.memory_bytes, frame_bytes * 3)
                self.assertEqual(streamer.target_indices(8, direction=1), (8, 9, 10))
                self.assertEqual(streamer.target_indices(8, direction=-1), (8, 7, 6))
                self.assertEqual(streamer.target_indices(8), (8, 7, 9))

    def test_interactive_seek_loads_target_first_and_emits_ready_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "interactive.tpdata",
                frame_count=12,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                store.positions[:] = np.arange(36, dtype=np.float32).reshape(12, 1, 3)
                store.flush()
                ready: list[int] = []
                streamer = FrameStreamer(
                    store,
                    prefetch_radius=5,
                    interactive_prefetch_frames=3,
                    frame_ready_callback=ready.append,
                )
                try:
                    streamer.start()
                    streamer.seek(7, direction=1, interactive=True)
                    self.assertIsNotNone(streamer.wait_for_frame(7, timeout_s=2.0))
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and not ready:
                        time.sleep(0.01)

                    self.assertEqual(ready, [7])
                    self.assertEqual(streamer.target_indices(7, direction=1, interactive=True), (7, 8, 9))
                finally:
                    streamer.stop()

    def test_frame_lease_prevents_slot_reuse_until_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "lease.tpdata",
                frame_count=2,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                store.positions[:] = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
                store.flush()
                streamer = FrameStreamer(
                    store,
                    prefetch_radius=0,
                    max_memory_bytes=12,
                )
                try:
                    streamer.start()
                    streamer.seek(0)
                    self.assertIsNotNone(streamer.wait_for_frame(0, timeout_s=2.0))
                    lease = streamer.acquire_frame(0)
                    self.assertIsNotNone(lease)
                    expected = lease.positions.copy()

                    streamer.seek(1)
                    time.sleep(0.05)
                    self.assertFalse(streamer.has_frame(1))
                    np.testing.assert_array_equal(lease.positions, expected)

                    lease.release()
                    lease.release()
                    self.assertIsNotNone(streamer.wait_for_frame(1, timeout_s=2.0))
                    self.assertTrue(streamer.has_frame(1))
                finally:
                    streamer.stop()

    def test_background_prefetch_uses_bounded_ring_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "stream.tpdata",
                frame_count=10,
                atom_numbers=np.array([1, 6], dtype=np.uint16),
                symbols=["H", "C"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
                store_cells=True,
            ) as store:
                store.positions[:] = np.arange(60, dtype=np.float32).reshape(10, 2, 3)
                store.cells[:] = np.arange(90, dtype=np.float32).reshape(10, 3, 3)
                store.flush()

                streamer = FrameStreamer(store, prefetch_radius=2)
                try:
                    streamer.start()
                    streamer.seek(4)
                    frame = streamer.wait_for_frame(4, timeout_s=2.0)
                    self.assertIsNotNone(frame)
                    np.testing.assert_array_equal(frame, store.frame(4))
                    np.testing.assert_array_equal(streamer.get_cell(4), store.cell(4))

                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        cached = streamer.cached_indices()
                        if set(cached) == {2, 3, 4, 5, 6}:
                            break
                        time.sleep(0.01)

                    self.assertEqual(set(streamer.cached_indices()), {2, 3, 4, 5, 6})
                    self.assertLessEqual(len(streamer.cached_indices()), 5)
                    self.assertEqual(streamer.memory_bytes, (5 * 2 * 3 * 4) + (5 * 3 * 3 * 4))
                finally:
                    streamer.stop()

    def test_seek_inside_same_prefetch_window_does_not_restart_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "stream.tpdata",
                frame_count=8,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                streamer = FrameStreamer(store, prefetch_radius=20)
                streamer.seek(0)
                generation = streamer._generation

                streamer.seek(5)

                self.assertEqual(streamer._generation, generation)

    def test_window_ready_reports_when_prefetch_window_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "ready.tpdata",
                frame_count=6,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                store.positions[:] = np.arange(18, dtype=np.float32).reshape(6, 1, 3)
                store.flush()
                streamer = FrameStreamer(store, prefetch_radius=2)
                try:
                    self.assertEqual(streamer.target_window_count(3), 5)
                    self.assertFalse(streamer.is_window_ready(3))
                    streamer.start()
                    streamer.seek(3)
                    self.assertIsNotNone(streamer.wait_for_frame(3, timeout_s=2.0))
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and not streamer.is_window_ready(3):
                        time.sleep(0.01)
                    self.assertTrue(streamer.is_window_ready(3))
                finally:
                    streamer.stop()

    def test_loader_errors_are_propagated_and_wake_waiters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "error.tpdata",
                frame_count=2,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                errors: list[BaseException] = []

                def failed_frame(_frame_index: int) -> np.ndarray:
                    raise OSError("simulated frame read failure")

                store.frame = failed_frame  # type: ignore[method-assign]
                streamer = FrameStreamer(store, error_callback=errors.append)
                streamer.start()
                streamer.seek(0)

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not errors:
                    time.sleep(0.01)

                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], OSError)
                self.assertIs(streamer.error, errors[0])
                self.assertIsNone(streamer.wait_for_frame(0, timeout_s=0.1))
                self.assertTrue(streamer.stop())

    def test_stop_keeps_thread_owned_until_blocking_read_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BinaryTrajectoryStore.create(
                Path(tmp) / "blocking.tpdata",
                frame_count=1,
                atom_numbers=np.array([1], dtype=np.uint16),
                symbols=["H"],
                source_path=None,
                source_mtime_ns=0,
                source_size=0,
            ) as store:
                entered = threading.Event()
                release = threading.Event()
                original_frame = store.frame

                def blocking_frame(frame_index: int) -> np.ndarray:
                    entered.set()
                    release.wait(timeout=2.0)
                    return original_frame(frame_index)

                store.frame = blocking_frame  # type: ignore[method-assign]
                streamer = FrameStreamer(store)
                streamer.start()
                streamer.seek(0)
                self.assertTrue(entered.wait(timeout=1.0))

                self.assertFalse(streamer.stop(timeout_s=0.01))
                self.assertTrue(streamer.is_alive)
                release.set()
                self.assertTrue(streamer.stop(timeout_s=1.0))
                self.assertFalse(streamer.is_alive)


if __name__ == "__main__":
    unittest.main()
