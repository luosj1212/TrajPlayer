from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


GROUP_ORDER = (
    "Qt",
    "Chemfiles",
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
    if "chemfiles" in parts or "chemfiles" in name:
        return "Chemfiles"
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


def check_bundle_growth(
    report: dict[str, object],
    *,
    baseline_bytes: int,
    max_growth_percent: float,
) -> int:
    if baseline_bytes <= 0:
        raise ValueError("baseline_bytes must be positive")
    if max_growth_percent < 0.0:
        raise ValueError("max_growth_percent must be non-negative")
    limit = int(baseline_bytes * (1.0 + max_growth_percent / 100.0))
    total = int(report["total_bytes"])
    if total > limit:
        growth = ((total / baseline_bytes) - 1.0) * 100.0
        raise RuntimeError(
            f"Portable bundle grew {growth:.2f}% to {total} bytes; "
            f"the allowed limit is {limit} bytes"
        )
    return limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Report portable bundle size by dependency")
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--max-growth-percent", type=float, default=5.0)
    args = parser.parse_args()

    report = bundle_size_report(args.root or discover_bundle_root())
    print(f"Bundle: {report['root']}")
    print(f"Files: {report['file_count']}")
    for group, size in report["groups"].items():
        print(f"{group:18} {size / (1024.0 * 1024.0):8.2f} MiB")
    print(f"{'Total':18} {report['total_bytes'] / (1024.0 * 1024.0):8.2f} MiB")
    if args.baseline_file is not None:
        if not args.platform:
            parser.error("--platform is required with --baseline-file")
        baselines = json.loads(args.baseline_file.read_text(encoding="utf-8"))
        try:
            baseline_bytes = int(baselines["platforms"][args.platform])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"No bundle-size baseline for platform {args.platform!r}"
            ) from exc
        limit = check_bundle_growth(
            report,
            baseline_bytes=baseline_bytes,
            max_growth_percent=args.max_growth_percent,
        )
        print(
            f"Baseline ({args.platform}) {baseline_bytes / (1024.0 * 1024.0):.2f} MiB; "
            f"limit {limit / (1024.0 * 1024.0):.2f} MiB"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
