from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
REQUIRED_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
)
FORBIDDEN_PATH_SUFFIXES = (".tar.gz", ".tpdata", ".traj", ".trr", ".xtc", ".zip")
FORBIDDEN_TEXT = (
    "E:" + "\\Anaconda",
    "C:" + "\\Users\\" + "15804",
    "C:" + "/Users/" + "15804",
)


def release_files() -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        publishable_roots = (
            ROOT / ".github",
            ROOT / "docs",
            ROOT / "examples",
            ROOT / "scripts",
            ROOT / "tests",
            ROOT / "trajplayer",
        )
        files = [
            path
            for base in publishable_roots
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.endswith(".tpdata") for part in path.parts)
        ]
        files.extend(
            path
            for path in ROOT.iterdir()
            if path.is_file()
            and (
                path.name in {".gitattributes", ".gitignore", "LICENSE"}
                or path.suffix.lower()
                in {".bat", ".md", ".py", ".sh", ".spec", ".toml", ".txt"}
            )
        )
        return sorted(set(files))
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required release file: {relative}")

    files = release_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 10 MiB: {relative}")
        if relative.lower().endswith(FORBIDDEN_PATH_SUFFIXES):
            errors.append(f"generated trajectory/archive would be tracked: {relative}")
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        normalized_source = source.replace("\\\\", "\\")
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in normalized_source:
                errors.append(f"local machine path in {relative}: {forbidden}")

    init_source = (ROOT / "trajplayer" / "__init__.py").read_text(encoding="utf-8")
    version_match = re.search(r'__version__\s*=\s*"([^"]+)"', init_source)
    display_match = re.search(r'__display_version__\s*=\s*"([^"]+)"', init_source)
    if version_match is None or display_match is None:
        errors.append("TrajPlayer version metadata is missing")
    else:
        package_version = version_match.group(1)
        display_version = display_match.group(1)
        if package_version != display_version.replace("-alpha.", "a"):
            errors.append("package and display versions do not match")
        release_notes = ROOT / "docs" / "releases" / f"v{display_version}.md"
        if not release_notes.is_file():
            errors.append(f"missing release notes: {release_notes.relative_to(ROOT)}")

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for requirement in requirements:
        requirement = requirement.strip()
        if requirement and not requirement.startswith("#") and "==" not in requirement:
            errors.append(f"runtime dependency is not pinned: {requirement}")

    if errors:
        raise SystemExit("Release check failed:\n- " + "\n- ".join(errors))
    print(f"Release check passed for {len(files)} publishable files.")


if __name__ == "__main__":
    main()
