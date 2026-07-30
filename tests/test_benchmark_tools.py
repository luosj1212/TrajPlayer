import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajplayer.benchmark_store import create_synthetic_store
from trajplayer.benchmark_stats import BenchmarkDiagnostics
from trajplayer.render_stats import RenderStats


class BenchmarkToolTests(unittest.TestCase):
    def test_benchmark_diagnostics_summarizes_pipeline_cadence(self) -> None:
        diagnostics = BenchmarkDiagnostics()

        for index in range(4):
            timestamp = 20.0 + index / 60.0
            diagnostics.record_tick(timestamp_s=timestamp)
            diagnostics.record_decision(timestamp_s=timestamp, dropped_frames=index % 2)
            diagnostics.record_upload(timestamp_s=timestamp)
        diagnostics.record_no_frame()
        diagnostics.record_duplicate_frame()

        summary = diagnostics.summary()
        self.assertEqual(summary["timer_ticks"], 4)
        self.assertEqual(summary["playback_decisions"], 4)
        self.assertEqual(summary["uploaded_frames"], 4)
        self.assertEqual(summary["dropped_frames_total"], 2)
        self.assertEqual(summary["no_frame_count"], 1)
        self.assertEqual(summary["duplicate_frame_count"], 1)
        self.assertAlmostEqual(summary["timer_hz"], 60.0)
        self.assertAlmostEqual(summary["decision_hz"], 60.0)
        self.assertAlmostEqual(summary["upload_hz"], 60.0)

    def test_render_stats_tracks_average_p95_and_draw_call_invariant(self) -> None:
        stats = RenderStats()

        for index, paint_ms in enumerate((8.0, 9.0, 11.0, 7.0)):
            stats.record_frame(
                paint_ms=paint_ms,
                upload_ms=1.5,
                draw_calls=1,
                timestamp_s=10.0 + index / 60.0,
            )

        summary = stats.summary()
        self.assertEqual(summary["frames"], 4)
        self.assertAlmostEqual(summary["paint_ms_avg"], 8.75)
        self.assertEqual(summary["paint_ms_max"], 11.0)
        self.assertEqual(summary["paint_ms_p95"], 11.0)
        self.assertEqual(summary["draw_calls_max"], 1)
        self.assertTrue(summary["single_draw_call_per_frame"])
        self.assertAlmostEqual(summary["frame_span_s"], 3 / 60.0)
        self.assertAlmostEqual(summary["cadence_fps"], 60.0)
        self.assertGreater(summary["render_budget_fps_avg"], 90.0)

    def test_synthetic_store_creates_contiguous_float32_memmap_without_python_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "synthetic.tpdata"
            with create_synthetic_store(root, frame_count=5, atom_count=7, chunk_frames=2) as store:
                self.assertEqual(store.positions.shape, (5, 7, 3))
                self.assertEqual(store.positions.dtype, np.float32)
                self.assertIsInstance(store.positions, np.memmap)
                self.assertEqual(store.atom_count, 7)
                self.assertEqual(store.frame_count, 5)
                self.assertEqual(store.metadata["source"]["size"], 5 * 7 * 3 * 4)
                self.assertTrue(store.frame(3).flags["C_CONTIGUOUS"])
                self.assertFalse(np.allclose(store.frame(0), store.frame(3)))


if __name__ == "__main__":
    unittest.main()
