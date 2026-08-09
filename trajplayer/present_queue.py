from __future__ import annotations


class FramePresentQueue:
    """A one-deep queue acknowledged only after QOpenGLWidget swaps a frame."""

    def __init__(self) -> None:
        self._pending_frame: int | None = None

    @property
    def pending_frame(self) -> int | None:
        return self._pending_frame

    @property
    def has_pending_frame(self) -> bool:
        return self._pending_frame is not None

    def begin(self, frame_index: int) -> bool:
        if self._pending_frame is not None:
            return False
        self._pending_frame = int(frame_index)
        return True

    def acknowledge(self) -> int | None:
        frame_index = self._pending_frame
        self._pending_frame = None
        return frame_index

    def clear(self) -> int | None:
        return self.acknowledge()
