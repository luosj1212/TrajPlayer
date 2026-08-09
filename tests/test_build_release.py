import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.build_release import (
    _platform_label,
    validate_portable_archive,
    validate_portable_tree,
)


WINDOWS_MEMBERS = (
    "TrajPlayer/TrajPlayer.exe",
    "TrajPlayer/_internal/python310.dll",
    "TrajPlayer/_internal/numpy/_core/_multiarray_umath.cp310-win_amd64.pyd",
    "TrajPlayer/_internal/numpy.libs/openblas.dll",
    "TrajPlayer/_internal/trajplayer/_trajcore.cp310-win_amd64.pyd",
)
MACOS_MEMBERS = (
    "TrajPlayer-macOS/TrajPlayer.app/Contents/MacOS/TrajPlayer",
    "TrajPlayer-macOS/TrajPlayer.app/Contents/Frameworks/numpy/_core/"
    "_multiarray_umath.cpython-311-darwin.so",
    "TrajPlayer-macOS/TrajPlayer.app/Contents/Frameworks/PySide6/Qt/plugins/"
    "platforms/libqcocoa.dylib",
    "TrajPlayer-macOS/TrajPlayer.app/Contents/Frameworks/trajplayer/"
    "_trajcore.cpython-311-darwin.so",
)


class BuildReleaseTests(unittest.TestCase):
    def test_validate_portable_tree_accepts_complete_windows_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist_root = Path(temporary_directory)
            for member in WINDOWS_MEMBERS:
                path = dist_root / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            validate_portable_tree(dist_root / "TrajPlayer", system="Windows")

    def test_validate_portable_tree_rejects_missing_numpy_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist_root = Path(temporary_directory)
            for member in WINDOWS_MEMBERS:
                if "_multiarray_umath" in member:
                    continue
                path = dist_root / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with self.assertRaisesRegex(SystemExit, "_multiarray_umath"):
                validate_portable_tree(dist_root / "TrajPlayer", system="Windows")

    def test_validate_portable_tree_rejects_missing_native_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist_root = Path(temporary_directory)
            for member in WINDOWS_MEMBERS:
                if "_trajcore" in member:
                    continue
                path = dist_root / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with self.assertRaisesRegex(SystemExit, "_trajcore"):
                validate_portable_tree(dist_root / "TrajPlayer", system="Windows")

    def test_validate_portable_archive_checks_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "TrajPlayer.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for member in WINDOWS_MEMBERS:
                    archive.writestr(member, b"")

            validate_portable_archive(archive_path, system="Windows")

    def test_validate_portable_tree_accepts_a_macos_app_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist_root = Path(temporary_directory)
            for member in MACOS_MEMBERS:
                path = dist_root / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            validate_portable_tree(
                dist_root / "TrajPlayer-macOS",
                system="macOS",
            )

    def test_validate_portable_archive_accepts_a_macos_app_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "TrajPlayer-macOS.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for member in MACOS_MEMBERS:
                    archive.writestr(member, b"")

            validate_portable_archive(archive_path, system="macOS")

    def test_macos_release_names_match_native_architecture(self) -> None:
        with (
            patch("scripts.build_release.platform.system", return_value="Darwin"),
            patch("scripts.build_release.platform.machine", return_value="arm64"),
        ):
            self.assertEqual(_platform_label(), ("macOS", "arm64", "ditto-zip"))


if __name__ == "__main__":
    unittest.main()
