from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QToolTip

from .model import TimelineModel


class TimelineWidget(QSlider):
    markerRequested = Signal(int)
    markerActivated = Signal(int)

    def __init__(self, model: TimelineModel, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._model = model
        self._accent = QColor("#1769aa")
        self._marker_color = QColor("#d9485f")
        self._range_color = QColor(47, 133, 90, 70)
        self._cursor_color = QColor("#7b4ab5")
        self.setMouseTracking(True)
        self.setMinimumHeight(28)
        model.changed.connect(self._on_model_changed)

    def set_colors(
        self,
        *,
        accent: str,
        marker: str,
        playback_range: str,
        cursor: str,
    ) -> None:
        self._accent = QColor(accent)
        self._marker_color = QColor(marker)
        self._range_color = QColor(playback_range)
        self._range_color.setAlpha(70)
        self._cursor_color = QColor(cursor)
        self.update()

    def frame_at_x(self, x: float) -> int:
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        span = max(1, groove.width())
        position = max(0, min(int(round(x - groove.left())), span))
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), position, span
        )

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self.maximum() <= self.minimum():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        if (
            self._model.range_start != self.minimum()
            or self._model.range_end != self.maximum()
        ):
            start_x = self._x_for_frame(self._model.range_start, groove)
            end_x = self._x_for_frame(self._model.range_end, groove)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._range_color)
            painter.drawRoundedRect(
                QRectF(start_x, groove.center().y() - 3, max(2.0, end_x - start_x), 6),
                3,
                3,
            )
        for marker in self._model.markers:
            x = int(round(self._x_for_frame(marker.frame_index, groove)))
            color = QColor(marker.color) if marker.color else self._marker_color
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygon([QPoint(x, 2), QPoint(x - 4, 8), QPoint(x + 4, 8)])
            )
        if self._model.analysis_cursor is not None:
            x = self._x_for_frame(self._model.analysis_cursor, groove)
            painter.setPen(QPen(self._cursor_color, 1.5))
            painter.drawLine(int(round(x)), 1, int(round(x)), self.height() - 2)
        if self._model.preview_frame is not None:
            x = self._x_for_frame(self._model.preview_frame, groove)
            painter.setPen(QPen(self._accent, 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(int(round(x)), 8, int(round(x)), self.height() - 2)
        painter.end()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.markerRequested.emit(self.frame_at_x(event.position().x()))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            marker = self._marker_near(event.position().x())
            if marker is not None:
                self.markerActivated.emit(marker.frame_index)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        marker = self._marker_near(event.position().x())
        if marker is not None and not event.buttons():
            QToolTip.showText(event.globalPosition().toPoint(), marker.label, self)
        super().mouseMoveEvent(event)

    def _on_model_changed(self) -> None:
        if self._model.frame_count > 0:
            self.setRange(0, self._model.frame_count - 1)
        else:
            self.setRange(0, 0)
        self.update()

    def _x_for_frame(self, frame: int, groove) -> float:
        span = max(1, groove.width())
        position = QStyle.sliderPositionFromValue(
            self.minimum(), self.maximum(), int(frame), span
        )
        return float(groove.left() + position)

    def _marker_near(self, x: float):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        for marker in self._model.markers:
            if abs(self._x_for_frame(marker.frame_index, groove) - x) <= 6.0:
                return marker
        return None
