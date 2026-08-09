from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from .diagnostics import collect_diagnostics
from .startup import error_log_path


class GuiSmokeController(QObject):
    """Exercise packaged trajectory loading and two real OpenGL frames."""

    def __init__(
        self,
        window,
        *,
        timeout_ms: int,
        output_path: Path | None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.timeout_s = max(1.0, int(timeout_ms) / 1000.0)
        self.output_path = output_path
        self.started_s = time.monotonic()
        self.stage = "first_frame"
        self.first_frame_metrics: dict[str, int] | None = None
        self.completed = False
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)
        self.window.automation_mode = True
        self.window.error_reported.connect(self.fail)

    def start(self) -> None:
        self.timer.start()

    def poll(self) -> None:
        if self.completed:
            return
        if time.monotonic() - self.started_s >= self.timeout_s:
            self.fail(f"GUI smoke test timed out during {self.stage}")
            return
        store = self.window.store
        gl_view = self.window.gl_view
        if store is None or self.window.displayed_frame < 0:
            return
        gl_info = gl_view.gl_diagnostics
        if not gl_view.isValid() or gl_info.get("version", "unknown") == "unknown":
            return

        if self.stage == "first_frame":
            if self.first_frame_metrics is None:
                try:
                    self.first_frame_metrics = framebuffer_metrics(
                        gl_view.grabFramebuffer()
                    )
                    _validate_framebuffer(self.first_frame_metrics)
                except Exception as exc:
                    self.fail(f"First OpenGL frame validation failed: {exc}")
                    return
            if store.navigable_frame_count < 2:
                if store.frame_count_is_final:
                    self.fail("GUI smoke trajectory must contain at least two frames")
                return
            self.stage = "second_frame"
            self.window.step_next()
            return

        if self.window.current_frame != 1 or self.window.displayed_frame != 1:
            return
        try:
            second_frame_metrics = framebuffer_metrics(gl_view.grabFramebuffer())
            _validate_framebuffer(second_frame_metrics)
        except Exception as exc:
            self.fail(f"Second OpenGL frame validation failed: {exc}")
            return
        self.succeed(second_frame_metrics)

    def succeed(self, second_frame_metrics: dict[str, int]) -> None:
        result = {
            "passed": True,
            "frames_rendered": [0, 1],
            "first_frame": self.first_frame_metrics,
            "second_frame": second_frame_metrics,
            "diagnostics": collect_diagnostics(
                opengl=self.window.gl_view.gl_diagnostics,
                log_path=error_log_path(),
            ),
        }
        self._finish(result, exit_code=0)

    def fail(self, message: str) -> None:
        if self.completed:
            return
        result = {
            "passed": False,
            "stage": self.stage,
            "error": str(message),
            "diagnostics": collect_diagnostics(
                opengl=self.window.gl_view.gl_diagnostics,
                log_path=error_log_path(),
            ),
        }
        self._finish(result, exit_code=2)

    def _finish(self, result: dict[str, object], *, exit_code: int) -> None:
        self.completed = True
        self.timer.stop()
        report = json.dumps(result, indent=2, sort_keys=True)
        print(f"[gui-smoke] {report}", flush=True)
        if self.output_path is not None:
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(report + "\n", encoding="utf-8")
            except OSError as exc:
                print(f"[gui-smoke] failed to write output: {exc}", flush=True)
                exit_code = 2
        self.window.close()
        app = QApplication.instance()
        if app is not None:
            app.exit(exit_code)


def framebuffer_metrics(image: QImage) -> dict[str, int]:
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    if width <= 0 or height <= 0:
        return {"width": width, "height": height, "non_background_pixels": 0}
    rows = np.frombuffer(converted.bits(), dtype=np.uint8, count=converted.sizeInBytes()).reshape(
        height,
        converted.bytesPerLine(),
    )
    rgba = rows[:, : width * 4].reshape(height, width, 4)
    background = rgba[0, 0, :3].astype(np.int16)
    color_delta = np.abs(rgba[:, :, :3].astype(np.int16) - background)
    non_background = np.any(color_delta > 8, axis=2)
    return {
        "width": width,
        "height": height,
        "non_background_pixels": int(np.count_nonzero(non_background)),
        "background_red": int(background[0]),
        "background_green": int(background[1]),
        "background_blue": int(background[2]),
    }


def _validate_framebuffer(metrics: dict[str, int]) -> None:
    if metrics["width"] <= 0 or metrics["height"] <= 0:
        raise RuntimeError("framebuffer has no pixels")
    if min(
        metrics["background_red"],
        metrics["background_green"],
        metrics["background_blue"],
    ) < 240:
        raise RuntimeError("framebuffer background is not the expected white")
    if metrics["non_background_pixels"] < 16:
        raise RuntimeError("framebuffer is blank or only contains the white background")
