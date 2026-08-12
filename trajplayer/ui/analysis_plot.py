from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from trajplayer.interaction.models import AnalysisResult


SERIES_COLORS = ("#1769aa", "#d9485f", "#2f855a", "#7b4ab5", "#c47a15")


def minmax_decimate(
    x: np.ndarray,
    y: np.ndarray,
    pixel_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.shape != x_values.shape:
        raise ValueError("x and y must be matching one-dimensional arrays")
    width = max(1, int(pixel_width))
    if x_values.size <= width * 2:
        return x_values, y_values
    edges = np.linspace(0, x_values.size, width + 1, dtype=np.int64)
    out_x: list[float] = []
    out_y: list[float] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        bucket = y_values[start:stop]
        finite = np.flatnonzero(np.isfinite(bucket))
        if finite.size == 0:
            continue
        low = start + int(finite[np.argmin(bucket[finite])])
        high = start + int(finite[np.argmax(bucket[finite])])
        for index in sorted({low, high}):
            out_x.append(float(x_values[index]))
            out_y.append(float(y_values[index]))
    return np.asarray(out_x), np.asarray(out_y)


def heatmap_values_for_range(
    result: AnalysisResult,
    x_limits: tuple[float, float],
) -> np.ndarray:
    if result.y.ndim != 2:
        raise ValueError("Heatmap data must be two-dimensional")
    low, high = (float(value) for value in x_limits)
    mask = (result.x >= low) & (result.x <= high)
    return np.asarray(result.y[mask], dtype=np.float64).T[::-1]


class AnalysisPlotWidget(QWidget):
    frameRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: AnalysisResult | None = None
        self._cursor_frame: int | None = None
        self._view_x: tuple[float, float] | None = None
        self._drag_start_x: float | None = None
        self._drag_current_x: float | None = None
        self._log_x = False
        self._log_y = False
        self._plot_cache: QImage | None = None
        self._empty_text = "No analysis result"
        self._background = QColor("#ffffff")
        self._text = QColor("#303740")
        self._grid = QColor("#dfe3e8")
        self._cursor = QColor("#7b4ab5")
        self._series_colors = tuple(QColor(color) for color in SERIES_COLORS)
        self.setMinimumHeight(170)
        self.setMouseTracking(True)

    @property
    def result(self) -> AnalysisResult | None:
        return self._result

    def set_result(self, result: AnalysisResult | None) -> None:
        self._result = result
        self._view_x = None
        self._cursor_frame = None
        self._plot_cache = None
        self.update()

    def set_cursor_frame(self, frame_index: int | None) -> None:
        value = None if frame_index is None else int(frame_index)
        if value == self._cursor_frame:
            return
        self._cursor_frame = value
        self.update()

    def set_colors(
        self,
        *,
        background: str,
        text: str,
        grid: str,
        cursor: str,
        series: tuple[str, ...] | None = None,
    ) -> None:
        self._background = QColor(background)
        self._text = QColor(text)
        self._grid = QColor(grid)
        self._cursor = QColor(cursor)
        if series:
            self._series_colors = tuple(QColor(color) for color in series)
        self._plot_cache = None
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = str(text)
        self.update()

    def reset_zoom(self) -> None:
        self._view_x = None
        self._plot_cache = None
        self.update()

    def set_log_axes(self, *, x: bool, y: bool) -> None:
        values = (bool(x), bool(y))
        if values == (self._log_x, self._log_y):
            return
        self._log_x, self._log_y = values
        self._plot_cache = None
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        result = self._result
        plot = self._plot_rect()
        if result is None or result.x.size == 0:
            painter.fillRect(self.rect(), self._background)
            painter.setPen(self._text)
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            painter.end()
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        expected_width = max(1, int(math.ceil(self.width() * dpr)))
        expected_height = max(1, int(math.ceil(self.height() * dpr)))
        if self._plot_cache is not None and (
            self._plot_cache.width() != expected_width
            or self._plot_cache.height() != expected_height
            or not math.isclose(
                float(self._plot_cache.devicePixelRatio()),
                dpr,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            self._plot_cache = None
        if self._plot_cache is None:
            self._plot_cache = self._render_plot_cache(result)
        painter.drawImage(QPointF(0.0, 0.0), self._plot_cache)
        self._draw_cursor(painter, plot, result)
        if self._drag_start_x is not None and self._drag_current_x is not None:
            left, right = sorted((self._drag_start_x, self._drag_current_x))
            painter.fillRect(QRectF(left, plot.top(), right - left, plot.height()), QColor(23, 105, 170, 45))
        painter.end()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        self._plot_cache = None
        super().resizeEvent(event)

    def _render_plot_cache(self, result: AnalysisResult) -> QImage:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        image = QImage(
            max(1, int(math.ceil(self.width() * dpr))),
            max(1, int(math.ceil(self.height() * dpr))),
            QImage.Format.Format_RGBA8888_Premultiplied,
        )
        image.setDevicePixelRatio(dpr)
        image.fill(self._background)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        plot = self._plot_rect()
        self._draw_axes(painter, plot, result)
        if bool(result.metadata.get("heatmap")) and result.y.ndim == 2:
            self._draw_heatmap(painter, plot, result)
        else:
            self._draw_lines(painter, plot, result)
        painter.end()
        return image

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self.reset_zoom()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._plot_rect().contains(event.position()):
            self._drag_start_x = float(event.position().x())
            self._drag_current_x = self._drag_start_x
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        result = self._result
        if self._drag_start_x is not None and event.buttons() & Qt.MouseButton.LeftButton:
            plot = self._plot_rect()
            self._drag_current_x = max(plot.left(), min(float(event.position().x()), plot.right()))
            self.update()
            return
        if result is None or not self._plot_rect().contains(event.position()):
            return
        index = self._nearest_index(float(event.position().x()), result)
        if index is None:
            return
        value = result.y[index]
        if np.ndim(value) == 0:
            value_text = f"{float(value):.6g} {result.y_unit}"
        else:
            value_text = ", ".join(f"{float(item):.5g}" for item in np.ravel(value)[:4])
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{result.x[index]:.6g} {result.x_unit}\n{value_text}",
            self,
        )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        result = self._result
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start_x is None:
            return
        start = self._drag_start_x
        end = self._drag_current_x if self._drag_current_x is not None else start
        self._drag_start_x = None
        self._drag_current_x = None
        if result is None:
            self.update()
            return
        if abs(end - start) >= 8.0:
            self._view_x = tuple(sorted((self._x_value(start, result), self._x_value(end, result))))
            self._plot_cache = None
        else:
            index = self._nearest_index(float(event.position().x()), result)
            if index is not None and result.metadata.get("x_kind") == "frame":
                frame_indices = result.metadata.get("frame_indices")
                frame = (
                    int(np.asarray(frame_indices)[index])
                    if frame_indices is not None
                    else int(round(float(result.x[index])))
                )
                self._cursor_frame = frame
                self.frameRequested.emit(frame)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(
            64.0,
            26.0,
            max(10.0, self.width() - 82.0),
            max(10.0, self.height() - 60.0),
        )

    def _x_limits(self, result: AnalysisResult) -> tuple[float, float]:
        if self._view_x is not None:
            return self._view_x
        source = result.x[result.x > 0.0] if self._log_x else result.x
        if source.size == 0:
            return (1.0, 10.0)
        low = float(np.nanmin(source))
        high = float(np.nanmax(result.x))
        return (low, high if high > low else low + 1.0)

    def _x_pixel(self, value: float, plot: QRectF, result: AnalysisResult) -> float:
        low, high = self._x_limits(result)
        if self._log_x:
            if value <= 0.0:
                return plot.left()
            value, low, high = math.log10(value), math.log10(low), math.log10(high)
        return plot.left() + (float(value) - low) / (high - low) * plot.width()

    def _x_value(self, pixel: float, result: AnalysisResult) -> float:
        plot = self._plot_rect()
        low, high = self._x_limits(result)
        fraction = (pixel - plot.left()) / max(1.0, plot.width())
        fraction = min(1.0, max(0.0, fraction))
        if self._log_x:
            return 10.0 ** (math.log10(low) + fraction * (math.log10(high) - math.log10(low)))
        return low + fraction * (high - low)

    def _nearest_index(self, pixel: float, result: AnalysisResult) -> int | None:
        if result.x.size == 0:
            return None
        value = self._x_value(pixel, result)
        index = int(np.searchsorted(result.x, value))
        candidates = [max(0, min(index, result.x.size - 1)), max(0, min(index - 1, result.x.size - 1))]
        return min(candidates, key=lambda item: abs(float(result.x[item]) - value))

    def _draw_axes(self, painter: QPainter, plot: QRectF, result: AnalysisResult) -> None:
        painter.setPen(QPen(self._grid, 1.0))
        for step in range(5):
            y = plot.top() + step * plot.height() / 4.0
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(self._text)
        x0, x1 = self._x_limits(result)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 5, 100, 20), f"{x0:.5g}")
        painter.drawText(QRectF(plot.right() - 100, plot.bottom() + 5, 100, 20), Qt.AlignmentFlag.AlignRight, f"{x1:.5g} {result.x_unit}")
        painter.drawText(QRectF(plot.left(), 2, 160, 20), result.y_unit)

    def _draw_lines(self, painter: QPainter, plot: QRectF, result: AnalysisResult) -> None:
        values = result.y[:, None] if result.y.ndim == 1 else result.y
        if self._log_y:
            transformed = np.full(values.shape, np.nan, dtype=np.float64)
            np.log10(values, out=transformed, where=values > 0.0)
            values = transformed
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        y_min, y_max = float(finite.min()), float(finite.max())
        if not math.isfinite(y_min) or not math.isfinite(y_max):
            return
        if y_max <= y_min:
            padding = max(1.0, abs(y_min)) * 0.05
            y_min -= padding
            y_max += padding
        x0, x1 = self._x_limits(result)
        mask = (result.x >= x0) & (result.x <= x1)
        x_visible = result.x[mask]
        for series in range(values.shape[1]):
            y_visible = values[mask, series]
            x_draw, y_draw = minmax_decimate(x_visible, y_visible, int(plot.width()))
            if x_draw.size == 0:
                continue
            path = QPainterPath()
            started = False
            for x_value, y_value in zip(x_draw, y_draw, strict=True):
                if not np.isfinite(y_value):
                    started = False
                    continue
                point = QPointF(
                    self._x_pixel(float(x_value), plot, result),
                    plot.bottom() - (float(y_value) - y_min) / (y_max - y_min) * plot.height(),
                )
                if not started:
                    path.moveTo(point)
                    started = True
                else:
                    path.lineTo(point)
            painter.setPen(
                QPen(
                    self._series_colors[series % len(self._series_colors)],
                    1.5,
                )
            )
            painter.drawPath(path)
        painter.setPen(self._text)
        shown_max = 10.0 ** y_max if self._log_y else y_max
        shown_min = 10.0 ** y_min if self._log_y else y_min
        painter.drawText(QRectF(4, plot.top(), 50, 20), Qt.AlignmentFlag.AlignRight, f"{shown_max:.4g}")
        painter.drawText(QRectF(4, plot.bottom() - 20, 50, 20), Qt.AlignmentFlag.AlignRight, f"{shown_min:.4g}")

    def _draw_heatmap(self, painter: QPainter, plot: QRectF, result: AnalysisResult) -> None:
        values = heatmap_values_for_range(result, self._x_limits(result))
        if values.size == 0:
            return
        if self._log_y:
            transformed = np.full(values.shape, np.nan, dtype=np.float64)
            np.log10(values, out=transformed, where=values > 0.0)
            values = transformed
        finite = np.isfinite(values)
        if not np.any(finite):
            return
        low, high = np.nanpercentile(values[finite], [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        rgba = np.empty((*normalized.shape, 4), dtype=np.uint8)
        rgba[..., 0] = np.clip(255 * (1.5 * normalized - 0.25), 0, 255)
        rgba[..., 1] = np.clip(255 * (1.5 - np.abs(normalized - 0.5) * 2.0), 0, 255)
        rgba[..., 2] = np.clip(255 * (1.25 - 1.5 * normalized), 0, 255)
        rgba[..., 3] = 255
        image = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()
        painter.drawImage(plot, image)

    def _draw_cursor(self, painter: QPainter, plot: QRectF, result: AnalysisResult) -> None:
        if self._cursor_frame is None or result.metadata.get("x_kind") != "frame":
            return
        frame_indices = result.metadata.get("frame_indices")
        if frame_indices is not None:
            frame_values = np.asarray(frame_indices, dtype=np.int64)
            index = int(np.searchsorted(frame_values, self._cursor_frame))
            index = max(0, min(index, frame_values.size - 1))
            if index > 0 and abs(int(frame_values[index - 1]) - self._cursor_frame) < abs(int(frame_values[index]) - self._cursor_frame):
                index -= 1
            cursor_value = float(result.x[index])
        else:
            cursor_value = float(self._cursor_frame)
        x0, x1 = self._x_limits(result)
        if not x0 <= cursor_value <= x1:
            return
        x = self._x_pixel(cursor_value, plot, result)
        painter.setPen(QPen(self._cursor, 1.4, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
