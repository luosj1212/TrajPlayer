import tempfile
import unittest
from pathlib import Path

from scripts.report_bundle_size import (
    bundle_size_report,
    check_bundle_growth,
    classify_bundle_member,
)


class BundleSizeReportTests(unittest.TestCase):
    def test_members_are_grouped_by_runtime_dependency(self) -> None:
        self.assertEqual(classify_bundle_member(Path("_internal/PySide6/Qt6Core.dll")), "Qt")
        self.assertEqual(
            classify_bundle_member(Path("_internal/chemfiles/chemfiles.dll")),
            "Chemfiles",
        )
        self.assertEqual(classify_bundle_member(Path("_internal/numpy/_core/test.pyd")), "NumPy")
        self.assertEqual(classify_bundle_member(Path("_internal/ase/io/traj.py")), "ASE")
        self.assertEqual(classify_bundle_member(Path("LICENSES/THIRD_PARTY_LICENSES.txt")), "Licenses and docs")
        self.assertEqual(classify_bundle_member(Path("TrajPlayer.exe")), "TrajPlayer")

    def test_report_totals_real_files_without_following_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            qt_file = root / "_internal" / "PySide6" / "Qt6Core.dll"
            numpy_file = root / "_internal" / "numpy" / "core.pyd"
            qt_file.parent.mkdir(parents=True)
            numpy_file.parent.mkdir(parents=True)
            qt_file.write_bytes(b"q" * 10)
            numpy_file.write_bytes(b"n" * 7)

            report = bundle_size_report(root)

            self.assertEqual(report["file_count"], 2)
            self.assertEqual(report["total_bytes"], 17)
            self.assertEqual(report["groups"]["Qt"], 10)
            self.assertEqual(report["groups"]["NumPy"], 7)

    def test_growth_check_rejects_more_than_five_percent(self) -> None:
        with self.assertRaises(RuntimeError):
            check_bundle_growth(
                {"total_bytes": 106},
                baseline_bytes=100,
                max_growth_percent=5.0,
            )
        self.assertEqual(
            check_bundle_growth(
                {"total_bytes": 105},
                baseline_bytes=100,
                max_growth_percent=5.0,
            ),
            105,
        )


if __name__ == "__main__":
    unittest.main()
