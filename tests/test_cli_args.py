import unittest
from pathlib import Path

from trajplayer.cli_args import parse_cli_args


class CliArgsTests(unittest.TestCase):
    def test_parse_benchmark_args(self) -> None:
        args = parse_cli_args(
            [
                "--startup-smoke",
                "--native-smoke",
                "--gui-smoke",
                "--gui-smoke-output=C:/tmp/gui-smoke.json",
                "--gui-smoke-timeout-ms=20000",
                "--reader-smoke=C:/tmp/fixtures",
                "--reader-smoke-output=C:/tmp/readers.json",
                "--doctor-output=C:/tmp/doctor.json",
                "--smoke-exit-ms=500",
                "--benchmark-output=C:/tmp/out.json",
                "--benchmark-root=C:/tmp/bench.tpdata",
                "--benchmark-atoms=100000",
                "--benchmark-frames=64",
                "--benchmark-render-frames=120",
                "--benchmark-finish-gpu",
                "--benchmark-bonds",
                "--benchmark-mode=bond",
                "C:/tmp/source.traj",
            ]
        )

        self.assertTrue(args.startup_smoke)
        self.assertTrue(args.native_smoke)
        self.assertTrue(args.gui_smoke)
        self.assertEqual(args.gui_smoke_output, Path("C:/tmp/gui-smoke.json"))
        self.assertEqual(args.gui_smoke_timeout_ms, 20000)
        self.assertEqual(args.reader_smoke, Path("C:/tmp/fixtures"))
        self.assertEqual(args.reader_smoke_output, Path("C:/tmp/readers.json"))
        self.assertEqual(args.doctor_output, Path("C:/tmp/doctor.json"))
        self.assertEqual(args.smoke_exit_ms, 500)
        self.assertEqual(args.benchmark_output, Path("C:/tmp/out.json"))
        self.assertEqual(args.benchmark_root, Path("C:/tmp/bench.tpdata"))
        self.assertEqual(args.benchmark_atoms, 100000)
        self.assertEqual(args.benchmark_frames, 64)
        self.assertEqual(args.benchmark_render_frames, 120)
        self.assertTrue(args.benchmark_finish_gpu)
        self.assertTrue(args.benchmark_bonds)
        self.assertEqual(args.benchmark_mode, "bond")
        self.assertEqual(args.paths, [Path("C:/tmp/source.traj")])


if __name__ == "__main__":
    unittest.main()
