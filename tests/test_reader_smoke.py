import tempfile
import unittest
from pathlib import Path

from scripts.create_smoke_fixtures import create_smoke_fixtures
from trajplayer.reader_smoke import SMOKE_CASES, run_reader_smoke, write_reader_smoke_report


class ReaderSmokeTests(unittest.TestCase):
    def test_all_portable_reader_cases_open_first_and_last_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_smoke_fixtures(Path(tmp) / "fixtures")
            report = run_reader_smoke(root)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), len(SMOKE_CASES))

            output = Path(tmp) / "reader-smoke.json"
            write_reader_smoke_report(report, output)
            self.assertIn('"passed": true', output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
