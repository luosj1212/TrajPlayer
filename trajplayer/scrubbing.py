from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SliderScrubState:
    preview_interval_s: float = 1.0 / 30.0
    active: bool = False
    pending_frame: int | None = None
    last_preview_s: float | None = None
    previewed_frame: int | None = None

    def begin(self, frame_index: int) -> int:
        self.active = True
        self.pending_frame = int(frame_index)
        self.last_preview_s = None
        self.previewed_frame = None
        return self.pending_frame

    def move(self, frame_index: int) -> int:
        self.pending_frame = int(frame_index)
        return self.pending_frame

    def preview_due(self, now_s: float) -> bool:
        if not self.active or self.pending_frame is None:
            return False
        if self.pending_frame == self.previewed_frame:
            return False
        if self.last_preview_s is None:
            return True
        return float(now_s) - self.last_preview_s >= self.preview_interval_s

    def mark_preview(self, now_s: float) -> int | None:
        if self.pending_frame is None:
            return None
        self.last_preview_s = float(now_s)
        self.previewed_frame = int(self.pending_frame)
        return self.previewed_frame

    def release(self, frame_index: int) -> int:
        frame = int(frame_index)
        self.active = False
        self.pending_frame = None
        self.last_preview_s = None
        self.previewed_frame = None
        return frame

    def should_commit_value_change(self) -> bool:
        return not self.active
