from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from trajplayer.analysis.pbc import minimum_image_displacement
from trajplayer.interaction.measurements import Measurement, MeasurementValue


@dataclass(frozen=True)
class MeasurementOverlayEntry:
    measurement_id: UUID
    atom_indices: tuple[int, ...]
    world_positions: np.ndarray
    label: str
    draft: bool = False


def measurement_display_positions(
    measurement: Measurement,
    positions: np.ndarray,
    cell: np.ndarray | None,
) -> np.ndarray:
    frame = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(measurement.atom_indices, dtype=np.int64)
    selected = np.ascontiguousarray(frame[indices], dtype=np.float64)
    if measurement.pbc_mode != "minimum_image" or cell is None:
        return selected
    if selected.shape[0] == 2:
        selected[1] = selected[0] + minimum_image_displacement(
            selected[1] - selected[0], cell
        )
    elif selected.shape[0] == 3:
        selected[0] = selected[1] + minimum_image_displacement(
            selected[0] - selected[1], cell
        )
        selected[2] = selected[1] + minimum_image_displacement(
            selected[2] - selected[1], cell
        )
    else:
        selected[0] = selected[1] + minimum_image_displacement(
            selected[0] - selected[1], cell
        )
        selected[2] = selected[1] + minimum_image_displacement(
            selected[2] - selected[1], cell
        )
        selected[3] = selected[2] + minimum_image_displacement(
            frame[indices[3]] - frame[indices[2]], cell
        )
    return selected


def overlay_entry(
    value: MeasurementValue,
    positions: np.ndarray,
    cell: np.ndarray | None,
    *,
    draft: bool = False,
) -> MeasurementOverlayEntry:
    suffix = "A" if value.unit == "A" else "deg"
    return MeasurementOverlayEntry(
        measurement_id=value.measurement.measurement_id,
        atom_indices=value.measurement.atom_indices,
        world_positions=measurement_display_positions(value.measurement, positions, cell),
        label=f"{value.value:.3f} {suffix}",
        draft=bool(draft),
    )


class ViewportOverlay(QWidget):
    """Paint only the handful of active measurement annotations."""

    def __init__(self, viewport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._entries: tuple[MeasurementOverlayEntry, ...] = ()
        self._line_color = QColor("#d9485f")
        self._draft_color = QColor("#1769aa")
        self._text_color = QColor("#20242a")
        self._label_background = QColor(255, 255, 255, 225)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_entries(self, entries: tuple[MeasurementOverlayEntry, ...]) -> None:
        self._entries = tuple(entries)
        self.update()

    def set_colors(
        self,
        *,
        line: str,
        draft: str,
        text: str,
        background: str,
    ) -> None:
        self._line_color = QColor(line)
        self._draft_color = QColor(draft)
        self._text_color = QColor(text)
        self._label_background = QColor(background)
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        if not self._entries:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for entry in self._entries:
            points, visible = self._viewport.project_world_positions(entry.world_positions)
            if len(points) < 2 or not np.any(visible):
                continue
            color = self._draft_color if entry.draft else self._line_color
            pen = QPen(color, 1.8 if entry.draft else 2.2)
            pen.setStyle(Qt.PenStyle.DashLine if entry.draft else Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.drawPath(path)
            painter.setBrush(color)
            for point, atom_index in zip(points, entry.atom_indices, strict=True):
                painter.drawEllipse(point, 3.8, 3.8)
                painter.drawText(point + QPointF(6.0, -5.0), str(atom_index + 1))
            anchor = QPointF(
                sum(point.x() for point in points) / len(points),
                sum(point.y() for point in points) / len(points),
            )
            self._draw_label(painter, anchor, entry.label, color)
        painter.end()

    def _draw_label(
        self,
        painter: QPainter,
        anchor: QPointF,
        label: str,
        border: QColor,
    ) -> None:
        metrics = QFontMetrics(painter.font())
        bounds = metrics.boundingRect(label)
        rect = QRectF(
            anchor.x() - bounds.width() * 0.5 - 6.0,
            anchor.y() - bounds.height() - 10.0,
            bounds.width() + 12.0,
            bounds.height() + 6.0,
        )
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(self._label_background)
        painter.drawRoundedRect(rect, 4.0, 4.0)
        painter.setPen(self._text_color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
