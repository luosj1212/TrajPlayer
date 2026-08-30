import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trajplayer.startup import format_numpy_import_error, missing_numpy_runtime_components


class StartupTests(unittest.TestCase):
    def test_importing_app_does_not_install_runtime_process_hooks(self) -> None:
        sys.modules.pop("app", None)
        with patch("trajplayer.startup.initialize_runtime") as initialize:
            importlib.import_module("app")
        initialize.assert_not_called()

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

    def test_macos_numpy_runtime_only_requires_the_bundled_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(
                missing_numpy_runtime_components(root, system="Darwin"),
                ("numpy/_core/_multiarray_umath*.so",),
            )
            extension = root / "numpy" / "_core" / "_multiarray_umath.cpython-311-darwin.so"
            extension.parent.mkdir(parents=True)
            extension.touch()

            self.assertEqual(
                missing_numpy_runtime_components(root, system="Darwin"),
                (),
            )

    def test_macos_import_error_uses_macos_recovery_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            message = format_numpy_import_error(
                ImportError("native extension failed"),
                Path(temporary_directory),
                system="Darwin",
            )

        self.assertIn("macOS ZIP", message)
        self.assertIn("TrajPlayer.app", message)
        self.assertNotIn("Linux tar.gz", message)


if __name__ == "__main__":
    unittest.main()
