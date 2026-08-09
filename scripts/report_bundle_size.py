from __future__ import annotations

import argparse
import hashlib
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
DYNAMIC_LIBRARY_SUFFIXES = frozenset({".dll", ".pyd", ".dylib", ".so"})
DEFAULT_TOP_FILES = 20
DEFAULT_GROUP_WARNING_BYTES = 5 * 1024 * 1024
FORBIDDEN_PORTABLE_PACKAGES = frozenset({"mdanalysis", "scipy"})


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
    if (
        name in {"trajplayer", "trajplayer.exe"}
        or "trajplayer.app" in parts
        or "trajplayer" in parts
    ):
        return "TrajPlayer"
    return "Other"


def bundle_size_report(root: Path, *, top_n: int = DEFAULT_TOP_FILES) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Portable bundle directory not found: {root}")
    groups: dict[str, int] = defaultdict(int)
    members: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative_path = path.relative_to(root)
        size = path.stat().st_size
        group = classify_bundle_member(relative_path)
        groups[group] += size
        members.append(
            {
                "path": relative_path.as_posix(),
                "size_bytes": int(size),
                "group": group,
                "_absolute_path": path,
            }
        )
    ordered_groups = {group: int(groups.get(group, 0)) for group in GROUP_ORDER}
    top_files = sorted(members, key=lambda member: int(member["size_bytes"]), reverse=True)
    dynamic_libraries = [member for member in top_files if _is_dynamic_library(str(member["path"]))]
    duplicates = _duplicate_members(members)
    forbidden_dependencies = sorted(
        str(member["path"])
        for member in members
        if FORBIDDEN_PORTABLE_PACKAGES.intersection(
            parts_part.lower()
            for parts_part in Path(str(member["path"])).parts
        )
    )

    def public_members(items: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in items[: max(0, int(top_n))]
        ]

    return {
        "root": str(root),
        "file_count": len(members),
        "total_bytes": int(sum(ordered_groups.values())),
        "groups": ordered_groups,
        "top_files": public_members(top_files),
        "top_dynamic_libraries": public_members(dynamic_libraries),
        "duplicate_bytes": int(sum(item["wasted_bytes"] for item in duplicates)),
        "duplicates": duplicates,
        "forbidden_dependencies": forbidden_dependencies,
    }


def _is_dynamic_library(relative_path: str) -> bool:
    name = Path(relative_path).name.lower()
    return Path(name).suffix in DYNAMIC_LIBRARY_SUFFIXES or ".so." in name


def _duplicate_members(members: list[dict[str, object]]) -> list[dict[str, object]]:
    by_size: dict[int, list[dict[str, object]]] = defaultdict(list)
    for member in members:
        by_size[int(member["size_bytes"])].append(member)
    duplicates: list[dict[str, object]] = []
    for size, candidates in by_size.items():
        if size <= 0 or len(candidates) < 2:
            continue
        by_digest: dict[str, list[dict[str, object]]] = defaultdict(list)
        for member in candidates:
            digest = _sha256_file(Path(member["_absolute_path"]))
            by_digest[digest].append(member)
        for digest, matches in by_digest.items():
            if len(matches) < 2:
                continue
            paths = sorted(str(match["path"]) for match in matches)
            duplicates.append(
                {
                    "sha256": digest,
                    "size_bytes": size,
                    "copies": len(paths),
                    "wasted_bytes": size * (len(paths) - 1),
                    "paths": paths,
                }
            )
    return sorted(duplicates, key=lambda item: int(item["wasted_bytes"]), reverse=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def bundle_group_deltas(
    report: dict[str, object],
    baseline_groups: dict[str, object],
) -> dict[str, int]:
    report_groups = report.get("groups", {})
    if not isinstance(report_groups, dict):
        raise ValueError("report groups must be an object")
    return {
        group: int(report_groups.get(group, 0)) - int(baseline_groups.get(group, 0))
        for group in GROUP_ORDER
    }


def platform_baseline(
    baselines: dict[str, object],
    platform: str,
) -> tuple[int, dict[str, object]]:
    platforms = baselines.get("platforms")
    if not isinstance(platforms, dict) or platform not in platforms:
        raise KeyError(platform)
    entry = platforms[platform]
    if isinstance(entry, (int, float)):
        return int(entry), {}
    if not isinstance(entry, dict):
        raise TypeError(platform)
    groups = entry.get("groups", {})
    if not isinstance(groups, dict):
        raise TypeError(f"{platform}.groups")
    return int(entry["total_bytes"]), groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Report portable bundle size by dependency")
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--max-growth-percent", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_FILES)
    args = parser.parse_args()

    report = bundle_size_report(args.root or discover_bundle_root(), top_n=args.top)
    print(f"Bundle: {report['root']}")
    print(f"Files: {report['file_count']}")
    for group, size in report["groups"].items():
        print(f"{group:18} {size / (1024.0 * 1024.0):8.2f} MiB")
    print(f"{'Total':18} {report['total_bytes'] / (1024.0 * 1024.0):8.2f} MiB")
    print("Largest files:")
    for member in report["top_files"]:
        print(f"  {member['size_bytes'] / (1024.0 * 1024.0):8.2f} MiB  {member['path']}")
    print(
        f"Duplicate payload: {report['duplicate_bytes'] / (1024.0 * 1024.0):.2f} MiB"
    )
    if args.baseline_file is not None:
        if not args.platform:
            parser.error("--platform is required with --baseline-file")
        baselines = json.loads(args.baseline_file.read_text(encoding="utf-8"))
        try:
            baseline_bytes, baseline_groups = platform_baseline(baselines, args.platform)
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
        for group, delta in bundle_group_deltas(report, baseline_groups).items():
            print(f"  {group:18} delta {delta / (1024.0 * 1024.0):+8.2f} MiB")
            if delta > DEFAULT_GROUP_WARNING_BYTES:
                print(
                    f"::warning title=Bundle group growth::{group} grew by "
                    f"{delta / (1024.0 * 1024.0):.2f} MiB"
                )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["forbidden_dependencies"]:
        raise SystemExit(
            "Forbidden portable dependencies were collected:\n- "
            + "\n- ".join(report["forbidden_dependencies"])
        )


if __name__ == "__main__":
    main()
