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
        self.assertIn("base_dir=portable_dir.name", release_source)
        self.assertIn('"gztar"', release_source)

    def test_macos_build_creates_a_native_app_bundle_and_preserves_it(self) -> None:
        spec_source = Path("TrajPlayer.spec").read_text(encoding="utf-8")
        build_source = Path("scripts/build_release.py").read_text(encoding="utf-8")
        script_source = Path("build_macos.sh").read_text(encoding="utf-8")

        self.assertIn("if sys.platform == 'darwin':", spec_source)
        self.assertIn("app = BUNDLE(", spec_source)
        self.assertIn("CFBundleDocumentTypes", spec_source)
        self.assertIn("io.github.luosj1212.TrajPlayer", spec_source)
        self.assertIn('"ditto"', build_source)
        self.assertIn('"codesign"', build_source)
        self.assertIn("python scripts/build_release.py", script_source)

    def test_wsl_build_uses_linux_filesystem_for_the_virtualenv(self) -> None:
        source = Path("build_linux_wsl.sh").read_text(encoding="utf-8")
        self.assertIn("mktemp -d", source)
        self.assertIn('bash "$build_root/build_linux.sh"', source)
        self.assertNotIn(".venv-linux/bin/python", source)

    def test_linux_runtime_does_not_modify_dll_search_path(self) -> None:
        source = Path("trajplayer/startup.py").read_text(encoding="utf-8")
        function = source.split("def configure_dll_search_path(", 1)[1]
        function = function.split("def missing_numpy_runtime_components(", 1)[0]
        self.assertIn('if os.name != "nt":', function)
        self.assertIn("return", function)

    def test_ci_and_release_exercise_real_gui_opengl_frames(self) -> None:
        ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        release = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

        self.assertIn("--gui-smoke", ci)
        self.assertIn("xvfb-run", ci)
        self.assertIn("--gui-smoke", release)
        self.assertIn("--reader-smoke", ci)
        self.assertIn("--reader-smoke", release)
        self.assertIn("xvfb-run", release)
        self.assertIn("libgl1-mesa-dri", release)
        self.assertIn("macos-15", ci)
        self.assertIn("QT_QPA_PLATFORM: cocoa", ci)
        self.assertIn("macOS-arm64", release)
        self.assertIn("macOS-x86_64", release)
        self.assertIn("Smoke-test macOS app bundle", release)
        self.assertIn("codesign --verify --deep --strict", release)


if __name__ == "__main__":
    unittest.main()
