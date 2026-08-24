from pathlib import Path

from trajplayer import __display_version__, __version__


ROOT = Path(__file__).resolve().parents[1]


def test_release_version() -> None:
    assert __version__ == "0.1.0a12"
    assert __display_version__ == "0.1.0-alpha.12"


def test_runtime_dependencies_are_reproducibly_pinned() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert requirements
    assert all("==" in line for line in requirements if line and not line.startswith("#"))


def test_release_scripts_have_no_developer_machine_paths() -> None:
    paths = (
        ROOT / "build_exe.bat",
        ROOT / "build_linux.sh",
        ROOT / "build_macos.sh",
        ROOT / "run_app.bat",
        ROOT / "TrajPlayer.spec",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "E:" + "\\Anaconda" not in source
    assert "C:\\Users\\" not in source


def test_pyproject_exposes_console_entrypoint() -> None:
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'trajplayer = "app:main"' in source
    assert 'license = "MIT"' in source
    assert '"Operating System :: MacOS :: MacOS X"' in source
