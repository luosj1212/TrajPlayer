from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


GROUP_ORDER = (
    "Qt",
    "MDAnalysis",
    "SciPy",
    "NumPy",
    "ASE",
    "TrajPlayer",
    "Licenses and docs",
    "Other",
)


def classify_bundle_member(relative_path: Path) -> str:
    parts = tuple(part.lower() for part in relative_path.parts)
    name = relative_path.name.lower()
    if any(part in {"pyside6", "shiboken6"} for part in parts) or "qt6" in name:
        return "Qt"
    if any(part in {"mdanalysis", "mda_xdrlib"} for part in parts):
        return "MDAnalysis"
    if any(part in {"scipy", "scipy.libs"} for part in parts):
        return "SciPy"
    if any(part in {"numpy", "numpy.libs"} for part in parts):
        return "NumPy"
    if "ase" in parts:
        return "ASE"
    if any(part in {"licenses", "docs"} for part in parts) or name.startswith(
        ("license", "readme", "third_party")
    ):
        return "Licenses and docs"
    if name in {"trajplayer", "trajplayer.exe"} or "trajplayer.app" in parts:
        return "TrajPlayer"
    return "Other"


def bundle_size_report(root: Path) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Portable bundle directory not found: {root}")
    groups: dict[str, int] = defaultdict(int)
    file_count = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        groups[classify_bundle_member(path.relative_to(root))] += path.stat().st_size
        file_count += 1
    ordered_groups = {group: int(groups.get(group, 0)) for group in GROUP_ORDER}
    return {
        "root": str(root),
        "file_count": file_count,
        "total_bytes": int(sum(ordered_groups.values())),
        "groups": ordered_groups,
    }


def discover_bundle_root() -> Path:
    candidates = (Path("dist/TrajPlayer-macOS"), Path("dist/TrajPlayer"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("No built TrajPlayer bundle was found under dist/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report portable bundle size by dependency")
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = bundle_size_report(args.root or discover_bundle_root())
    print(f"Bundle: {report['root']}")
    print(f"Files: {report['file_count']}")
    for group, size in report["groups"].items():
        print(f"{group:18} {size / (1024.0 * 1024.0):8.2f} MiB")
    print(f"{'Total':18} {report['total_bytes'] / (1024.0 * 1024.0):8.2f} MiB")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
