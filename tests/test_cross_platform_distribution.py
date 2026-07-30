from pathlib import Path
import unittest


class CrossPlatformDistributionTests(unittest.TestCase):
    def test_pyinstaller_spec_has_no_fixed_project_path(self) -> None:
        source = Path("TrajPlayer.spec").read_text(encoding="utf-8")
        self.assertIn("Path(SPECPATH)", source)
        self.assertIn("if os.name == 'nt':", source)
        self.assertNotIn("C:\\\\Users\\\\", source)
        self.assertNotIn("E:" + "\\\\Anaconda", source)

    def test_pyinstaller_collects_only_used_qt_modules_and_excludes_notebook_stacks(self) -> None:
        source = Path("TrajPlayer.spec").read_text(encoding="utf-8")
        self.assertNotIn("collect_dynamic_libs('PySide6')", source)
        self.assertIn("'PySide6.QtOpenGLWidgets'", source)
        self.assertIn("'matplotlib'", source)
        self.assertIn("'IPython'", source)

    def test_linux_build_script_creates_an_onedir_archive(self) -> None:
        source = Path("build_linux.sh").read_text(encoding="utf-8")
        self.assertIn("python scripts/build_release.py", source)
        release_source = Path("scripts/build_release.py").read_text(encoding="utf-8")
        self.assertIn('base_dir="TrajPlayer"', release_source)
        self.assertIn('"gztar"', release_source)

    def test_wsl_build_uses_linux_filesystem_for_the_virtualenv(self) -> None:
        source = Path("build_linux_wsl.sh").read_text(encoding="utf-8")
        self.assertIn("mktemp -d", source)
        self.assertIn('bash "$build_root/build_linux.sh"', source)
        self.assertNotIn(".venv-linux/bin/python", source)

    def test_linux_runtime_does_not_modify_dll_search_path(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")
        function = source.split("def configure_dll_search_path() -> None:", 1)[1]
        function = function.split("def enable_high_resolution_timers()", 1)[0]
        self.assertIn('if os.name != "nt":', function)
        self.assertIn("return", function)


if __name__ == "__main__":
    unittest.main()
