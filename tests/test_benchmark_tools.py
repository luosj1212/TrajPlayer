import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajplayer.benchmark_store import create_synthetic_store
from trajplayer.benchmark_stats import BenchmarkDiagnostics
from trajplayer.process_memory import process_memory_snapshot
from trajplayer.render_stats import RenderStats
from trajplayer.telemetry import RollingLatency
from scripts.compare_perf import compare_performance
from scripts.check_perf_report import performance_report_errors


class BenchmarkToolTests(unittest.TestCase):
    def test_benchmark_diagnostics_summarizes_pipeline_cadence(self) -> None:
        diagnostics = BenchmarkDiagnostics()

        for index in range(4):
            timestamp = 20.0 + index / 60.0
            diagnostics.record_tick(timestamp_s=timestamp)
            diagnostics.record_decision(timestamp_s=timestamp, dropped_frames=index % 2)
            diagnostics.record_upload(timestamp_s=timestamp)
            diagnostics.record_present_latency(1.0 + index)
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
        self.assertEqual(summary["present_latency_ms_p50"], 2.0)
        self.assertEqual(summary["present_latency_ms_p95"], 4.0)
        self.assertEqual(summary["present_latency_ms_p99"], 4.0)

    def test_render_stats_tracks_average_p95_and_draw_call_invariant(self) -> None:
        stats = RenderStats()

        for index, paint_ms in enumerate((8.0, 9.0, 11.0, 7.0)):
            stats.record_frame(
                paint_ms=paint_ms,
                upload_ms=1.5,
                draw_calls=1,
                timestamp_s=10.0 + index / 60.0,
                depth_sort_ms=float(index),
            )

        summary = stats.summary()
        self.assertEqual(summary["frames"], 4)
        self.assertAlmostEqual(summary["paint_ms_avg"], 8.75)
        self.assertEqual(summary["paint_ms_max"], 11.0)
        self.assertEqual(summary["paint_ms_p95"], 11.0)
        self.assertEqual(summary["paint_ms_p50"], 8.0)
        self.assertEqual(summary["paint_ms_p99"], 11.0)
        self.assertEqual(summary["upload_ms_p50"], 1.5)
        self.assertEqual(summary["upload_ms_p95"], 1.5)
        self.assertEqual(summary["upload_ms_p99"], 1.5)
        self.assertEqual(summary["depth_sort_ms_p95"], 3.0)
        self.assertEqual(summary["renderer_full_frame_copy_bytes"], 0)
        self.assertEqual(summary["draw_calls_max"], 1)
        self.assertTrue(summary["single_draw_call_per_frame"])
        self.assertAlmostEqual(summary["frame_span_s"], 3 / 60.0)
        self.assertAlmostEqual(summary["cadence_fps"], 60.0)
        self.assertGreater(summary["render_budget_fps_avg"], 90.0)

    def test_rolling_latency_keeps_a_fixed_sample_window(self) -> None:
        latency = RollingLatency(capacity=3)
        for value in (1.0, 2.0, 3.0, 100.0):
            latency.record(value)

        self.assertEqual(latency.total_count, 4)
        self.assertEqual(latency.sample_count, 3)
        self.assertEqual(latency.percentile(50), 3.0)
        self.assertEqual(latency.percentile(99), 100.0)
        self.assertEqual(latency.span(), 98.0)

    def test_render_stats_remain_bounded_during_long_playback(self) -> None:
        stats = RenderStats()
        for index in range(5000):
            stats.record_frame(
                paint_ms=1.0,
                upload_ms=0.25,
                draw_calls=1,
                timestamp_s=index / 60.0,
            )

        summary = stats.summary()
        self.assertEqual(summary["frames"], 5000)
        self.assertEqual(stats.paint_ms.sample_count, 4096)
        self.assertEqual(stats.timestamps_s.sample_count, 4096)
        self.assertAlmostEqual(summary["cadence_fps"], 60.0)

    def test_process_memory_snapshot_reports_current_process(self) -> None:
        snapshot = process_memory_snapshot()

        self.assertGreater(snapshot.rss_bytes, 0)
        self.assertGreaterEqual(snapshot.peak_rss_bytes, snapshot.rss_bytes)

    def test_perf_comparison_checks_relative_budget_and_invariants(self) -> None:
        baseline = {
            "render": {"cadence_fps": 60.0, "paint_ms_p95": 3.0},
            "pipeline": {"dropped_frames_total": 0},
            "copies": {"renderer_full_frame_copy_bytes": 0},
        }
        current = {
            "render": {
                "cadence_fps": 50.0,
                "paint_ms_p95": 3.4,
                "single_draw_call_per_frame": True,
            },
            "pipeline": {
                "dropped_frames_total": 1,
                "duplicate_frames": 1,
            },
            "copies": {"renderer_full_frame_copy_bytes": 0},
        }

        regressions = compare_performance(
            baseline,
            current,
            fail_regression_percent=10.0,
        )

        self.assertEqual(
            {regression.metric for regression in regressions},
            {
                "render.cadence_fps",
                "render.paint_ms_p95",
                "pipeline.dropped_frames_total",
                "pipeline.duplicate_frames",
            },
        )

    def test_perf_report_schema_rejects_missing_metrics(self) -> None:
        errors = performance_report_errors({"timed_out": False})

        self.assertTrue(any("startup.process_to_qapplication_ms" in error for error in errors))
        self.assertIn("render.frames must be positive", errors)

    def test_perf_report_rejects_duplicate_presentations(self) -> None:
        errors = performance_report_errors(
            {"pipeline": {"duplicate_frames": 1}, "timed_out": False}
        )

        self.assertIn("playback presented duplicate trajectory frames", errors)

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
