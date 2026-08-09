import tempfile
import unittest
from pathlib import Path

from scripts.report_bundle_size import (
    bundle_group_deltas,
    bundle_size_report,
    check_bundle_growth,
    classify_bundle_member,
    platform_baseline,
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
            self.assertEqual(report["top_files"][0]["path"], "_internal/PySide6/Qt6Core.dll")
            self.assertEqual(
                report["top_dynamic_libraries"][0]["path"],
                "_internal/PySide6/Qt6Core.dll",
            )

    def test_report_finds_duplicate_payload_without_following_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "a.dll"
            second = root / "nested" / "b.dll"
            second.parent.mkdir()
            first.write_bytes(b"same payload")
            second.write_bytes(b"same payload")

            report = bundle_size_report(root)

            self.assertEqual(report["duplicate_bytes"], len(b"same payload"))
            self.assertEqual(report["duplicates"][0]["copies"], 2)
            self.assertEqual(report["duplicates"][0]["paths"], ["a.dll", "nested/b.dll"])

    def test_report_flags_scipy_and_mdanalysis_package_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scipy_file = root / "_internal" / "scipy" / "sparse" / "core.pyc"
            mda_file = root / "_internal" / "MDAnalysis" / "lib" / "util.pyc"
            scipy_file.parent.mkdir(parents=True)
            mda_file.parent.mkdir(parents=True)
            scipy_file.touch()
            mda_file.touch()

            report = bundle_size_report(root)

            self.assertEqual(
                report["forbidden_dependencies"],
                [
                    "_internal/MDAnalysis/lib/util.pyc",
                    "_internal/scipy/sparse/core.pyc",
                ],
            )

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

    def test_structured_platform_baseline_and_group_deltas(self) -> None:
        baseline_bytes, groups = platform_baseline(
            {
                "platforms": {
                    "test": {
                        "total_bytes": 100,
                        "groups": {"Qt": 60, "Other": 40},
                    }
                }
            },
            "test",
        )

        self.assertEqual(baseline_bytes, 100)
        self.assertEqual(groups["Qt"], 60)
        deltas = bundle_group_deltas(
            {"groups": {"Qt": 65, "Other": 37}},
            groups,
        )
        self.assertEqual(deltas["Qt"], 5)
        self.assertEqual(deltas["Other"], -3)


if __name__ == "__main__":
    unittest.main()
