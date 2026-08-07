import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_release import validate_portable_archive, validate_portable_tree


WINDOWS_MEMBERS = (
    "TrajPlayer/TrajPlayer.exe",
    "TrajPlayer/_internal/python310.dll",
    "TrajPlayer/_internal/numpy/_core/_multiarray_umath.cp310-win_amd64.pyd",
    "TrajPlayer/_internal/numpy.libs/openblas.dll",
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

    def test_validate_portable_archive_checks_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "TrajPlayer.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for member in WINDOWS_MEMBERS:
                    archive.writestr(member, b"")

            validate_portable_archive(archive_path, system="Windows")


if __name__ == "__main__":
    unittest.main()
