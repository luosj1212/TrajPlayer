import json
import tempfile
import unittest
from pathlib import Path

from trajplayer.diagnostics import collect_diagnostics, diagnostics_json
from trajplayer.startup import redact_path, rotate_log


class DiagnosticsTests(unittest.TestCase):
    def test_home_paths_are_redacted(self) -> None:
        private_path = Path.home() / "private-project" / "trajectory.xtc"
        redacted = redact_path(private_path)

        self.assertIn("<home>", redacted)
        self.assertNotIn(Path.home().name, redacted)

    def test_report_includes_runtime_and_supplied_opengl_details(self) -> None:
        report = collect_diagnostics(
            opengl={"vendor": "Mesa", "renderer": "llvmpipe", "version": "4.5"}
        )

        self.assertIn("trajplayer", report)
        self.assertIn("packages", report)
        self.assertEqual(report["opengl"]["renderer"], "llvmpipe")
        json.loads(diagnostics_json(opengl=report["opengl"]))

    def test_large_logs_rotate_without_losing_the_previous_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "traj_player_error.log"
            first_backup = Path(tmp) / "traj_player_error.log.1"
            log_path.write_text("newest", encoding="utf-8")
            first_backup.write_text("older", encoding="utf-8")

            self.assertTrue(rotate_log(log_path, max_bytes=1, backups=2))
            self.assertEqual(first_backup.read_text(encoding="utf-8"), "newest")
            self.assertEqual(
                (Path(tmp) / "traj_player_error.log.2").read_text(encoding="utf-8"),
                "older",
            )


if __name__ == "__main__":
    unittest.main()
