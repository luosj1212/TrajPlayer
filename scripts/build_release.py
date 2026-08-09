from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from fnmatch import fnmatch
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = ROOT / "dist"
APP_DIR = DIST_ROOT / "TrajPlayer"
MAC_APP_DIR = DIST_ROOT / "TrajPlayer.app"
MAC_PACKAGE_DIR = DIST_ROOT / "TrajPlayer-macOS"

sys.path.insert(0, str(ROOT))

from scripts.collect_licenses import collect_licenses  # noqa: E402
from trajplayer import __display_version__  # noqa: E402


def _portable_runtime_patterns(system: str) -> tuple[str, ...]:
    if system == "Windows":
        return (
            "TrajPlayer/TrajPlayer.exe",
            "TrajPlayer/_internal/python*.dll",
            "TrajPlayer/_internal/numpy/_core/_multiarray_umath*.pyd",
            "TrajPlayer/_internal/numpy.libs/*.dll",
            "TrajPlayer/**/_trajcore*.pyd",
        )
    if system == "Linux":
        return (
            "TrajPlayer/TrajPlayer",
            "TrajPlayer/_internal/libpython3*.so*",
            "TrajPlayer/_internal/numpy/_core/_multiarray_umath*.so",
            "TrajPlayer/_internal/numpy.libs/*.so*",
            "TrajPlayer/**/_trajcore*.so",
        )
    if system in {"Darwin", "macOS"}:
        return (
            "TrajPlayer-macOS/TrajPlayer.app/Contents/MacOS/TrajPlayer",
            "TrajPlayer-macOS/TrajPlayer.app/Contents/Frameworks/*_multiarray_umath*.so",
            "TrajPlayer-macOS/TrajPlayer.app/Contents/Frameworks/*libqcocoa.dylib",
            "TrajPlayer-macOS/**/_trajcore*.so",
        )
    raise SystemExit(f"Unsupported release platform: {system}")


def _missing_patterns(members: set[str], patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern for pattern in patterns if not any(fnmatch(member, pattern) for member in members))


def validate_portable_tree(app_dir: Path = APP_DIR, *, system: str | None = None) -> None:
    system_name = system or platform.system()
    dist_root = app_dir.parent
    members = {
        path.relative_to(dist_root).as_posix()
        for path in app_dir.rglob("*")
        if path.is_file()
    }
    missing = _missing_patterns(members, _portable_runtime_patterns(system_name))
    if missing:
        raise SystemExit(f"Portable runtime is incomplete; missing: {', '.join(missing)}")


def validate_portable_archive(archive_path: Path, *, system: str | None = None) -> None:
    system_name = system or platform.system()
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            members = {name.replace("\\", "/") for name in archive.namelist() if not name.endswith("/")}
    else:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name.replace("\\", "/") for member in archive.getmembers() if member.isfile()}
    missing = _missing_patterns(members, _portable_runtime_patterns(system_name))
    if missing:
        raise SystemExit(f"Portable archive is incomplete; missing: {', '.join(missing)}")


def _platform_label() -> tuple[str, str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        architecture = "x64" if system == "Windows" else "x86_64"
    else:
        architecture = machine or "unknown"
    if system == "Windows":
        return system, architecture, "zip"
    if system == "Linux":
        return system, architecture, "gztar"
    if system == "Darwin":
        return "macOS", architecture, "ditto-zip"
    raise SystemExit(
        "Release packages are currently supported on Windows, Linux, and macOS only."
    )


def _prepare_portable_tree(system: str) -> Path:
    if system != "macOS":
        if not APP_DIR.is_dir():
            raise SystemExit(f"PyInstaller output was not created: {APP_DIR}")
        portable_dir = APP_DIR
    else:
        if not MAC_APP_DIR.is_dir():
            raise SystemExit(f"PyInstaller app bundle was not created: {MAC_APP_DIR}")
        subprocess.run(
            ["codesign", "--verify", "--deep", "--strict", str(MAC_APP_DIR)],
            check=True,
        )
        if MAC_PACKAGE_DIR.exists():
            shutil.rmtree(MAC_PACKAGE_DIR)
        MAC_PACKAGE_DIR.mkdir(parents=True)
        subprocess.run(
            ["ditto", str(MAC_APP_DIR), str(MAC_PACKAGE_DIR / "TrajPlayer.app")],
            check=True,
        )
        subprocess.run(
            [
                "codesign",
                "--verify",
                "--deep",
                "--strict",
                str(MAC_PACKAGE_DIR / "TrajPlayer.app"),
            ],
            check=True,
        )
        portable_dir = MAC_PACKAGE_DIR

    shutil.copy2(ROOT / "LICENSE", portable_dir / "LICENSE")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", portable_dir / "THIRD_PARTY_NOTICES.md")
    shutil.copy2(ROOT / "DISTRIBUTION_README.txt", portable_dir / "DISTRIBUTION_README.txt")
    collect_licenses(portable_dir / "licenses")
    return portable_dir


def _stage_native_extension() -> Path | None:
    spec = find_spec("trajplayer._trajcore")
    if spec is None or spec.origin is None:
        raise SystemExit("trajplayer._trajcore is not installed; build the native extension first")
    source = Path(spec.origin).resolve()
    if not source.is_file():
        raise SystemExit(f"trajplayer._trajcore was not found at {source}")
    target = (ROOT / "trajplayer" / source.name).resolve()
    if source == target:
        return None
    shutil.copy2(source, target)
    return target


def build_release(*, skip_pyinstaller: bool = False) -> Path:
    system, architecture, archive_format = _platform_label()
    if not skip_pyinstaller:
        staged_native = _stage_native_extension()
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "PyInstaller",
                    "--noconfirm",
                    "--clean",
                    str(ROOT / "TrajPlayer.spec"),
                ],
                cwd=ROOT,
                check=True,
            )
        finally:
            if staged_native is not None:
                staged_native.unlink(missing_ok=True)
    portable_dir = _prepare_portable_tree(system)
    validate_portable_tree(portable_dir, system=system)

    archive_base = ROOT / f"TrajPlayer-{system}-{architecture}-v{__display_version__}"
    archive_suffix = ".zip" if archive_format in {"zip", "ditto-zip"} else ".tar.gz"
    archive_path = Path(f"{archive_base}{archive_suffix}")
    archive_path.unlink(missing_ok=True)
    if archive_format == "ditto-zip":
        subprocess.run(
            [
                "ditto",
                "-c",
                "-k",
                "--sequesterRsrc",
                "--keepParent",
                str(portable_dir),
                str(archive_path),
            ],
            check=True,
        )
        created = archive_path
    else:
        created = Path(
            shutil.make_archive(
                str(archive_base),
                archive_format,
                root_dir=DIST_ROOT,
                base_dir=portable_dir.name,
            )
        )
    validate_portable_archive(created, system=system)
    digest = hashlib.sha256()
    with created.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum_path = Path(f"{created}.sha256")
    checksum_path.write_text(f"{digest.hexdigest()}  {created.name}\n", encoding="ascii")
    print(f"Release package created: {created}")
    print(f"SHA-256 written to: {checksum_path}")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable TrajPlayer release archive.")
    parser.add_argument(
        "--skip-pyinstaller",
        action="store_true",
        help="Archive an already completed dist/TrajPlayer build.",
    )
    args = parser.parse_args()
    build_release(skip_pyinstaller=args.skip_pyinstaller)


if __name__ == "__main__":
    main()
