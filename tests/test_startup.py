import tempfile
import unittest
from pathlib import Path

from trajplayer.startup import format_numpy_import_error, missing_numpy_runtime_components


class StartupTests(unittest.TestCase):
    def test_missing_numpy_runtime_components_reports_native_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            self.assertEqual(
                missing_numpy_runtime_components(root, system="Windows"),
                ("numpy/_core/_multiarray_umath*.pyd", "numpy.libs/*.dll"),
            )

    def test_complete_numpy_runtime_has_no_missing_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            extension = root / "numpy" / "_core" / "_multiarray_umath.cp310-win_amd64.pyd"
            library = root / "numpy.libs" / "openblas.dll"
            extension.parent.mkdir(parents=True)
            library.parent.mkdir(parents=True)
            extension.touch()
            library.touch()

            self.assertEqual(missing_numpy_runtime_components(root, system="Windows"), ())

    def test_import_error_explains_incomplete_portable_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            message = format_numpy_import_error(
                ModuleNotFoundError("numpy._core._multiarray_umath"),
                Path(temporary_directory),
                system="Windows",
            )

        self.assertIn("portable package is incomplete", message)
        self.assertIn("Do not run the EXE inside the ZIP", message)
        self.assertIn("Protection history", message)

    def test_linux_import_error_uses_linux_recovery_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            message = format_numpy_import_error(
                ImportError("native extension failed"),
                Path(temporary_directory),
                system="Linux",
            )

        self.assertIn("Linux tar.gz", message)
        self.assertNotIn("Windows Security", message)


if __name__ == "__main__":
    unittest.main()
