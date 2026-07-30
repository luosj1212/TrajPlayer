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
    args = parser.parse_args()

    QSurfaceFormat.setDefaultFormat(default_surface_format())
    application = QApplication([])
    window = TrajPlayerWindow()
    window.resize(1440, 920)
    window.show()

    trajectory = args.trajectory.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    def capture() -> None:
        image = window.grab()
        if not image.save(str(output), "PNG"):
            raise RuntimeError(f"Could not save screenshot to {output}")
        window.close()
        application.quit()

    QTimer.singleShot(100, lambda: window.load_trajectory_paths((trajectory,)))
    QTimer.singleShot(max(500, args.wait_ms), capture)
    application.exec()


if __name__ == "__main__":
    main()
