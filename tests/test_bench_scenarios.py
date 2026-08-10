import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.bench_scenarios import (
    _memory_slope_mib_min,
    _seek_indices,
    benchmark_open,
    benchmark_seek,
)
from trajplayer.trajectory_source import resolve_trajectory_source


class RealScenarioBenchmarkTests(unittest.TestCase):
    def test_open_and_seek_reports_use_a_real_extxyz_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = Path(tmp) / "c60.extxyz"
            shutil.copyfile(Path("examples/c60.extxyz"), trajectory)
            source = resolve_trajectory_source((trajectory,))

            opened = benchmark_open(
                source,
                repeat=1,
                wait_index=True,
                index_timeout_s=5.0,
            )
            sought = benchmark_seek(
                source,
                samples=5,
                pattern="storm",
                seed=1,
                wait_index=True,
                index_timeout_s=5.0,
            )

            self.assertEqual(opened["scenario"], "open")
            self.assertEqual(opened["trajectory"]["atom_count"], 60)
            self.assertEqual(opened["metrics"]["metadata_ms"]["count"], 1)
            self.assertEqual(sought["scenario"], "seek")
            self.assertEqual(sought["metrics"]["frame_read_ms"]["count"], 5)
            self.assertGreater(sought["metrics"]["decode_mib_s"], 0.0)

    def test_seek_patterns_and_memory_slope_are_deterministic(self) -> None:
        self.assertEqual(_seek_indices(5, sample_count=6, pattern="sequential", seed=0).tolist(), [0, 1, 2, 3, 4, 0])
        self.assertEqual(_seek_indices(5, sample_count=6, pattern="storm", seed=0).tolist(), [0, 4, 1, 3, 2, 2])
        self.assertAlmostEqual(
            _memory_slope_mib_min([(0.0, 100 * 1024 * 1024), (60.0, 110 * 1024 * 1024)]),
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
