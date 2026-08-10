from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


def tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            total += path.lstat().st_size
        elif path.is_file():
            total += path.stat().st_size
    return total


def elf_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        try:
            with path.open("rb") as handle:
                if handle.read(4) == b"\x7fELF":
                    result.append(path)
        except OSError:
            continue
    return result


def strip_tree(root: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    launcher = root / "TrajPlayer"
    paths = [
        path
        for path in elf_files(root)
        if path != launcher
        and not path.is_symlink()
        and "numpy.libs" not in path.relative_to(root).parts
    ]
    for path in paths:
        result = subprocess.run(
            ["strip", "--strip-unneeded", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"{path.relative_to(root)}: {result.stderr.strip()}")
    return len(paths), failures


def smoke(executable: Path, arguments: list[str], *, xvfb: bool = False) -> dict[str, object]:
    command = [str(executable), *arguments]
    if xvfb:
        command = ["xvfb-run", "-a", *command]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=executable.parent,
        env={**os.environ, "QT_LOGGING_RULES": "qt.qpa.*=false"},
    )
    report = {
        "arguments": arguments,
        "returncode": result.returncode,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "stdout": result.stdout.strip()[-1000:],
        "stderr": result.stderr.strip()[-1000:],
    }
    if result.returncode != 0:
        error_log = executable.parent / "traj_player_error.log"
        if error_log.is_file():
            report["error_log"] = error_log.read_text(
                encoding="utf-8",
                errors="replace",
            )[-4000:]
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def archive_tree(root: Path, output: Path) -> int:
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        archive.add(root, arcname=root.name)
    return output.stat().st_size


def smoke_bundle(
    root: Path,
    *,
    fixtures: Path | None,
    gui_trajectory: Path | None,
) -> list[dict[str, object]]:
    executable = root / "TrajPlayer"
    checks = [
        smoke(executable, ["--startup-smoke"]),
        smoke(executable, ["--native-smoke"]),
    ]
    if fixtures is not None:
        checks.append(smoke(executable, [f"--reader-smoke={fixtures.resolve()}"]))
    if gui_trajectory is not None:
        checks.append(
            smoke(
                executable,
                [
                    "--gui-smoke",
                    "--gui-smoke-timeout-ms=20000",
                    str(gui_trajectory.resolve()),
                ],
                xvfb=True,
            )
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test stripping a Linux portable bundle.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--gui-trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    with tempfile.TemporaryDirectory(prefix="trajplayer-strip-ab-") as temporary:
        workspace = Path(temporary)
        baseline_parent = workspace / "baseline"
        baseline_parent.mkdir()
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(baseline_parent)
        baseline = baseline_parent / "TrajPlayer"
        if not baseline.is_dir():
            raise SystemExit("archive does not contain TrajPlayer/")
        baseline_bytes = tree_bytes(baseline)
        variants = {
            "baseline": smoke_bundle(
                baseline,
                fixtures=args.fixtures,
                gui_trajectory=args.gui_trajectory,
            )
        }

        stripped = workspace / "stripped" / "TrajPlayer"
        shutil.copytree(baseline, stripped, symlinks=True)
        elf_count, failures = strip_tree(stripped)
        stripped_bytes = tree_bytes(stripped)
        variants["stripped"] = smoke_bundle(
            stripped,
            fixtures=args.fixtures,
            gui_trajectory=args.gui_trajectory,
        )
        stripped_archive_bytes = archive_tree(
            stripped,
            workspace / "TrajPlayer-stripped.tar.gz",
        )

        report = {
            "source_archive_bytes": archive.stat().st_size,
            "baseline_tree_bytes": baseline_bytes,
            "stripped_tree_bytes": stripped_bytes,
            "tree_saved_bytes": baseline_bytes - stripped_bytes,
            "tree_saved_percent": round(
                (baseline_bytes - stripped_bytes) * 100.0 / baseline_bytes,
                2,
            ),
            "stripped_archive_bytes": stripped_archive_bytes,
            "archive_saved_bytes": archive.stat().st_size - stripped_archive_bytes,
            "elf_files": elf_count,
            "strip_failures": failures,
            "smoke": variants,
        }
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.output is not None:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        if failures:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
