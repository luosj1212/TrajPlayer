from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class FrameMarker:
    frame_index: int
    label: str = ""
    color: str = "#d9485f"
    marker_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        frame = int(self.frame_index)
        if frame < 0:
            raise ValueError("frame_index must be non-negative")
        object.__setattr__(self, "frame_index", frame)
        object.__setattr__(self, "label", str(self.label))


class TimelineModel(QObject):
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.current_frame = 0
        self.preview_frame: int | None = None
        self.frame_count = 0
        self.frame_count_final = True
        self.range_start = 0
        self.range_end = 0
        self.markers: tuple[FrameMarker, ...] = ()
        self.analysis_cursor: int | None = None

    def reset(self, frame_count: int, *, final: bool = True) -> None:
        count = max(0, int(frame_count))
        self.current_frame = 0
        self.preview_frame = None
        self.frame_count = count
        self.frame_count_final = bool(final)
        self.range_start = 0
        self.range_end = max(0, count - 1)
        self.markers = ()
        self.analysis_cursor = None
        self.changed.emit()

    def set_frame_count(self, frame_count: int, *, final: bool) -> None:
        count = max(0, int(frame_count))
        old_last = max(0, self.frame_count - 1)
        range_was_full = self.range_start == 0 and self.range_end == old_last
        self.frame_count = count
        self.frame_count_final = bool(final)
        last = max(0, count - 1)
        self.current_frame = min(self.current_frame, last)
        if range_was_full:
            self.range_end = last
        else:
            self.range_start = min(self.range_start, last)
            self.range_end = max(self.range_start, min(self.range_end, last))
        self.markers = tuple(marker for marker in self.markers if marker.frame_index <= last)
        self.changed.emit()

    def set_current_frame(self, frame_index: int) -> None:
        frame = self._bounded(frame_index)
        if frame == self.current_frame:
            return
        self.current_frame = frame

    def set_preview_frame(self, frame_index: int | None) -> None:
        value = None if frame_index is None else self._bounded(frame_index)
        if value == self.preview_frame:
            return
        self.preview_frame = value
        self.changed.emit()

    def set_range(self, start: int, end: int) -> None:
        first = self._bounded(start)
        last = self._bounded(end)
        if first > last:
            first, last = last, first
        if (first, last) == (self.range_start, self.range_end):
            return
        self.range_start, self.range_end = first, last
        self.changed.emit()

    def add_marker(self, frame_index: int, label: str = "") -> FrameMarker:
        frame = self._bounded(frame_index)
        marker = FrameMarker(frame, label or f"Frame {frame + 1}")
        self.markers = tuple(
            sorted(self.markers + (marker,), key=lambda item: (item.frame_index, str(item.marker_id)))
        )
        self.changed.emit()
        return marker

    def remove_marker(self, marker_id: UUID) -> None:
        remaining = tuple(marker for marker in self.markers if marker.marker_id != marker_id)
        if remaining == self.markers:
            return
        self.markers = remaining
        self.changed.emit()

    def set_analysis_cursor(self, frame_index: int | None) -> None:
        value = None if frame_index is None else self._bounded(frame_index)
        if value == self.analysis_cursor:
            return
        self.analysis_cursor = value
        self.changed.emit()

    def _bounded(self, frame_index: int) -> int:
        if self.frame_count <= 0:
            return 0
        return max(0, min(int(frame_index), self.frame_count - 1))
