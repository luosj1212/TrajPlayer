from __future__ import annotations

import argparse
import shutil
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


RUNTIME_DISTRIBUTIONS = (
    "ase",
    "chemfiles",
    "numpy",
    "PySide6",
    "PySide6-Addons",
    "PySide6-Essentials",
    "shiboken6",
)
LICENSE_PREFIXES = ("license", "copying", "notice", "copyright")


def _license_files(distribution_name: str) -> list[Path]:
    try:
        package = distribution(distribution_name)
    except PackageNotFoundError:
        return []

    matches: list[Path] = []
    for entry in package.files or ():
        name = Path(entry).name.lower()
        parts = tuple(part.lower() for part in Path(entry).parts)
        in_metadata_licenses = "licenses" in parts and any(
            part.endswith((".dist-info", ".egg-info")) for part in parts
        )
        if not in_metadata_licenses and not name.startswith(LICENSE_PREFIXES):
            continue
        source = Path(package.locate_file(entry))
        if source.is_file() and source.stat().st_size <= 5 * 1024 * 1024:
            matches.append(source)
    return sorted(set(matches))


def collect_licenses(destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for distribution_name in RUNTIME_DISTRIBUTIONS:
        package_dir = destination / distribution_name.replace("-", "_")
        for index, source in enumerate(_license_files(distribution_name), start=1):
            package_dir.mkdir(parents=True, exist_ok=True)
            target_name = source.name if index == 1 else f"{index:02d}-{source.name}"
            shutil.copy2(source, package_dir / target_name)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect licenses for bundled runtime packages.")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = collect_licenses(args.destination.resolve())
    print(f"Collected {count} third-party license files in {args.destination.resolve()}")


if __name__ == "__main__":
    main()
