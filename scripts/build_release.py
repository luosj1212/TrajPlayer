from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = ROOT / "dist"
APP_DIR = DIST_ROOT / "TrajPlayer"

sys.path.insert(0, str(ROOT))

from scripts.collect_licenses import collect_licenses  # noqa: E402
from trajplayer import __display_version__  # noqa: E402


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
    raise SystemExit("Release packages are currently supported on Windows and Linux only.")


def build_release(*, skip_pyinstaller: bool = False) -> Path:
    system, architecture, archive_format = _platform_label()
    if not skip_pyinstaller:
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
    if not APP_DIR.is_dir():
        raise SystemExit(f"PyInstaller output was not created: {APP_DIR}")

    shutil.copy2(ROOT / "LICENSE", APP_DIR / "LICENSE")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", APP_DIR / "THIRD_PARTY_NOTICES.md")
    shutil.copy2(ROOT / "DISTRIBUTION_README.txt", APP_DIR / "DISTRIBUTION_README.txt")
    collect_licenses(APP_DIR / "licenses")

    archive_base = ROOT / f"TrajPlayer-{system}-{architecture}-v{__display_version__}"
    archive_suffix = ".zip" if archive_format == "zip" else ".tar.gz"
    archive_path = Path(f"{archive_base}{archive_suffix}")
    archive_path.unlink(missing_ok=True)
    created = Path(
        shutil.make_archive(
            str(archive_base),
            archive_format,
            root_dir=DIST_ROOT,
            base_dir="TrajPlayer",
        )
    )
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
