from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import TrajPlayerWindow
from trajplayer.gl_view import default_surface_format


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the current TrajPlayer README image.")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait-ms", type=int, default=3500)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=920)
    parser.add_argument("--language", choices=("en", "zh"))
    args = parser.parse_args()

    QSurfaceFormat.setDefaultFormat(default_surface_format())
    application = QApplication([])
    window = TrajPlayerWindow()
    previous_language = window.ui_language
    if args.language is not None and window.ui_language != args.language:
        index = window.language_combo.findData(args.language)
        window.language_combo.setCurrentIndex(index)
    window.resize(max(720, args.width), max(520, args.height))
    window.show()

    trajectory = args.trajectory.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    def capture() -> None:
        image = window.grab()
        if not image.save(str(output), "PNG"):
            raise RuntimeError(f"Could not save screenshot to {output}")
        if args.language is not None and args.language != previous_language:
            window._settings.setValue("ui/language", previous_language)
            window._settings.sync()
        window.close()
        application.quit()

    QTimer.singleShot(100, lambda: window.load_trajectory_paths((trajectory,)))
    QTimer.singleShot(max(500, args.wait_ms), capture)
    application.exec()


if __name__ == "__main__":
    main()
